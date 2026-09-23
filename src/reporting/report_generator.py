"""
Generate a downloadable PDF from already-computed pipeline results.

This module must not call run_pipeline() or recompute NLP, sentiment,
topics, drift, severity, or recommendations.
"""

from datetime import datetime
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

FONT_REGULAR = "Helvetica"
FONT_BOLD = "Helvetica-Bold"

from src.explainability.explainer import InsightExplainer


NAVY = colors.HexColor("#1f2937")
HEADER_BG = colors.HexColor("#1f2937")
ROW_ALT = colors.HexColor("#f3f4f6")
LINE = colors.HexColor("#d1d5db")
BODY = colors.HexColor("#111827")
MUTED = colors.HexColor("#4b5563")

DRIFT_COSINE_THRESHOLD = 0.80
CONCEPT_DRIFT_ALERT_THRESHOLD = 0.35
DRIFT_STATUS_ELEVATED = 0.35
DRIFT_STATUS_CRITICAL = 0.60


def generate_report(pipeline_result):
    """
    Build a PDF report from an existing run_pipeline() return value.

    Parameters
    ----------
    pipeline_result : dict
        The dict already produced by src.run_pipeline.run_pipeline.
        Must not be an uploaded file or raw CSV path.

    Returns
    -------
    bytes
        PDF file contents.
    """

    if not isinstance(pipeline_result, dict):
        raise ValueError(
            "generate_report() expects the stored pipeline result dict, "
            "not a file or DataFrame."
        )

    context = _build_context(pipeline_result)
    buffer = BytesIO()
    styles = _build_styles()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.75 * inch,
        title="Customer Feedback Intelligence Report",
        author="Customer Feedback Intelligence System",
    )

    story = []
    story.extend(_section_header(context, styles))
    story.extend(_section_executive(context, styles))
    story.extend(_section_attention(context, styles))
    story.extend(_section_sentiment(context, styles))
    story.extend(_section_prioritization(context, styles))
    story.extend(_section_drift(context, styles))
    story.extend(_section_topics(context, styles))
    story.extend(_section_sku(context, styles))
    story.extend(_section_recommendations(context, styles))
    story.extend(_section_insights(context, styles))
    story.extend(_section_methodology(styles))

    document.build(
        story,
        onFirstPage=_draw_page_footer,
        onLaterPages=_draw_page_footer,
    )

    return buffer.getvalue()


def _build_context(pipeline_result):
    processed_df = _copy_frame(pipeline_result.get("processed_df"))
    topics_df = _copy_frame(pipeline_result.get("topics_df"))
    aggregation = pipeline_result.get("aggregation_results") or {}
    sentiment_df = _copy_frame(aggregation.get("monthly_sentiment"))
    topic_frequency_df = _copy_frame(aggregation.get("topic_frequencies"))
    drift_results = pipeline_result.get("drift_results") or {}
    similarity_df = _copy_frame(drift_results.get("similarity_scores"))
    alerts = list(drift_results.get("alerts") or [])
    severity_df = _copy_frame(pipeline_result.get("severity_df"))
    recommendation_df = _copy_frame(
        pipeline_result.get("recommendation_df")
    )

    if not processed_df.empty and "date" in processed_df.columns:
        processed_df["date"] = pd.to_datetime(
            processed_df["date"],
            errors="coerce",
        )
        processed_df = processed_df.dropna(subset=["date"])

    if not processed_df.empty and "month" not in processed_df.columns:
        if "date" in processed_df.columns:
            processed_df["month"] = (
                processed_df["date"]
                .dt.to_period("M")
                .astype(str)
            )

    if (
        not processed_df.empty
        and "topic_id" in processed_df.columns
        and "topic_label" not in processed_df.columns
        and not topics_df.empty
    ):
        topic_mapping = dict(
            zip(
                topics_df["topic_id"],
                topics_df["topic_label"],
            )
        )
        processed_df["topic_label"] = (
            processed_df["topic_id"]
            .map(topic_mapping)
            .fillna("outlier")
        )

    latest_month = None
    if not processed_df.empty and "month" in processed_df.columns:
        latest_month = processed_df["month"].max()

    total_reviews = len(processed_df)

    valid_topics = 0
    if not processed_df.empty and "topic_id" in processed_df.columns:
        valid_topics = processed_df.loc[
            processed_df["topic_id"] != -1,
            "topic_id",
        ].nunique()

    latest_sentiment = np.nan
    latest_negative_percentage = np.nan
    if (
        latest_month is not None
        and "compound_score" in processed_df.columns
        and "sentiment_label" in processed_df.columns
    ):
        latest_df = processed_df[
            processed_df["month"] == latest_month
        ]
        if not latest_df.empty:
            latest_sentiment = latest_df["compound_score"].mean()
            latest_negative_percentage = (
                (latest_df["sentiment_label"] == "negative").mean()
                * 100
            )

    latest_severity_df = pd.DataFrame()
    highest_severity = 0
    if not severity_df.empty and latest_month is not None:
        latest_severity_df = severity_df[
            severity_df["month"] == latest_month
        ].copy()
        latest_severity_df = latest_severity_df.sort_values(
            "severity_score",
            ascending=False,
        )
        if not latest_severity_df.empty:
            highest_severity = float(
                latest_severity_df.iloc[0]["severity_score"]
            )

    latest_drift = 0
    drift_status = "Stable"
    if not similarity_df.empty:
        latest_drift_row = similarity_df.iloc[-1]
        latest_drift = float(
            latest_drift_row.get("concept_drift_score", 0) or 0
        )
        if latest_drift >= DRIFT_STATUS_CRITICAL:
            drift_status = "Critical Change"
        elif latest_drift >= DRIFT_STATUS_ELEVATED:
            drift_status = "Elevated Change"
        else:
            drift_status = "Stable"

    date_min = None
    date_max = None
    if not processed_df.empty and "date" in processed_df.columns:
        valid_dates = processed_df["date"].dropna()
        if not valid_dates.empty:
            date_min = valid_dates.min()
            date_max = valid_dates.max()

    explainer = InsightExplainer()
    sentiment_insight = _safe_explain(
        explainer.explain_sentiment,
        sentiment_df,
    )
    topic_insight = _safe_explain(
        explainer.explain_topic_growth,
        topic_frequency_df,
    )
    drift_insight = _safe_explain(
        explainer.explain_drift,
        similarity_df,
        DRIFT_COSINE_THRESHOLD,
    )

    sku_summary = pd.DataFrame()
    if not processed_df.empty and "sku" in processed_df.columns:
        sku_summary = (
            processed_df.groupby("sku")
            .agg(
                total_reviews=("review", "count"),
                avg_sentiment=("compound_score", "mean"),
                negative_reviews=(
                    "sentiment_label",
                    lambda x: (x == "negative").sum(),
                ),
            )
            .reset_index()
        )
        sku_summary["negative_percentage"] = (
            sku_summary["negative_reviews"]
            / sku_summary["total_reviews"]
            * 100
        )
        sku_summary = sku_summary.sort_values("avg_sentiment")

    attention_row = None
    if not recommendation_df.empty:
        attention_row = (
            recommendation_df
            .sort_values("severity_score", ascending=False)
            .iloc[0]
        )

    latest_recommendations = pd.DataFrame()
    if not recommendation_df.empty and latest_month is not None:
        latest_recommendations = (
            recommendation_df[
                recommendation_df["month"] == latest_month
            ]
            .sort_values("severity_score", ascending=False)
            .head(10)
        )

    return {
        "generated_at": datetime.now(),
        "processed_df": processed_df,
        "topics_df": topics_df,
        "sentiment_df": sentiment_df,
        "topic_frequency_df": topic_frequency_df,
        "similarity_df": similarity_df,
        "alerts": alerts,
        "severity_df": severity_df,
        "recommendation_df": recommendation_df,
        "latest_month": latest_month,
        "total_reviews": total_reviews,
        "valid_topics": valid_topics,
        "latest_sentiment": latest_sentiment,
        "latest_negative_percentage": latest_negative_percentage,
        "highest_severity": highest_severity,
        "latest_severity_df": latest_severity_df,
        "latest_drift": latest_drift,
        "drift_status": drift_status,
        "date_min": date_min,
        "date_max": date_max,
        "sentiment_insight": sentiment_insight,
        "topic_insight": topic_insight,
        "drift_insight": drift_insight,
        "sku_summary": sku_summary,
        "attention_row": attention_row,
        "latest_recommendations": latest_recommendations,
    }


def _copy_frame(value):
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return pd.DataFrame()


def _safe_explain(fn, *args):
    try:
        return fn(*args)
    except Exception:
        return None


def _build_styles():
    _register_fonts()
    font = FONT_REGULAR
    font_bold = FONT_BOLD
    base = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName=font_bold,
            fontSize=18,
            leading=22,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontName=font,
            fontSize=10,
            leading=13,
            textColor=MUTED,
            spaceAfter=10,
        ),
        "heading": ParagraphStyle(
            "ReportHeading",
            parent=base["Heading2"],
            fontName=font_bold,
            fontSize=13,
            leading=16,
            textColor=NAVY,
            spaceBefore=12,
            spaceAfter=8,
        ),
        "body": ParagraphStyle(
            "ReportBody",
            parent=base["Normal"],
            fontName=font,
            fontSize=9,
            leading=12,
            textColor=BODY,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        ),
        "cell": ParagraphStyle(
            "ReportCell",
            parent=base["Normal"],
            fontName=font,
            fontSize=8,
            leading=10,
            textColor=BODY,
        ),
        "cell_header": ParagraphStyle(
            "ReportCellHeader",
            parent=base["Normal"],
            fontName=font_bold,
            fontSize=8,
            leading=10,
            textColor=colors.white,
        ),
        "muted": ParagraphStyle(
            "ReportMuted",
            parent=base["Normal"],
            fontName=font,
            fontSize=8,
            leading=11,
            textColor=MUTED,
            spaceAfter=6,
        ),
        "footer": ParagraphStyle(
            "ReportFooter",
            parent=base["Normal"],
            fontName=font,
            fontSize=8,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
    }
    return styles


def _register_fonts():
    global FONT_REGULAR, FONT_BOLD

    if "ReportFont" in pdfmetrics.getRegisteredFontNames():
        FONT_REGULAR = "ReportFont"
        FONT_BOLD = "ReportFont-Bold"
        return

    regular, bold = _find_fonts()
    if regular is None:
        FONT_REGULAR = "Helvetica"
        FONT_BOLD = "Helvetica-Bold"
        return

    pdfmetrics.registerFont(TTFont("ReportFont", str(regular)))
    pdfmetrics.registerFont(TTFont("ReportFont-Bold", str(bold)))
    FONT_REGULAR = "ReportFont"
    FONT_BOLD = "ReportFont-Bold"


def _find_fonts():
    candidates = [
        (
            Path(r"C:\Windows\Fonts\arial.ttf"),
            Path(r"C:\Windows\Fonts\arialbd.ttf"),
        ),
        (
            Path(r"C:\Windows\Fonts\calibri.ttf"),
            Path(r"C:\Windows\Fonts\calibrib.ttf"),
        ),
        (
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ),
        (
            Path("/Library/Fonts/Arial.ttf"),
            Path("/Library/Fonts/Arial Bold.ttf"),
        ),
    ]

    for regular, bold in candidates:
        if regular.exists() and bold.exists():
            return regular, bold

    for regular, bold in candidates:
        if regular.exists():
            return regular, regular

    return None, None


def _draw_page_footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.4)
    canvas.line(
        doc.leftMargin,
        0.5 * inch,
        doc.pagesize[0] - doc.rightMargin,
        0.5 * inch,
    )
    canvas.setFont(FONT_REGULAR, 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(
        doc.leftMargin,
        0.32 * inch,
        "Customer Feedback Intelligence - decision-support report",
    )
    canvas.drawRightString(
        doc.pagesize[0] - doc.rightMargin,
        0.32 * inch,
        f"Page {doc.page}",
    )
    canvas.restoreState()


def _escape(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _fmt_number(value, digits=2):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_int(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_pct(value, already_percent=False):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    try:
        number = float(value)
        if not already_percent:
            number = number * 100
        return f"{number:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def _display_topic(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return str(value).replace("_", " ").title()


def _cell(text, styles, header=False):
    style = styles["cell_header"] if header else styles["cell"]
    return Paragraph(_escape(text), style)


def _make_table(headers, rows, styles, col_widths=None):
    data = [
        [_cell(h, styles, header=True) for h in headers]
    ]
    for row in rows:
        data.append([_cell(item, styles) for item in row])

    table = Table(
        data,
        colWidths=col_widths,
        repeatRows=1,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, 0), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.3, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, ROW_ALT],
                ),
            ]
        )
    )
    return table


def _chart_image(fig):
    buffer = BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=140,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    buffer.seek(0)
    image = Image(buffer)
    aspect = float(image.imageWidth) / float(image.imageHeight or 1)
    image.drawWidth = 6.9 * inch
    image.drawHeight = image.drawWidth / aspect
    return image


def _section_header(context, styles):
    generated = context["generated_at"].strftime("%Y-%m-%d %H:%M:%S")
    date_range = "Not available"
    if context["date_min"] is not None and context["date_max"] is not None:
        date_range = (
            f"{context['date_min'].strftime('%Y-%m-%d')} to "
            f"{context['date_max'].strftime('%Y-%m-%d')}"
        )

    period = context["latest_month"] or "N/A"

    return [
        Paragraph("Customer Feedback Intelligence", styles["title"]),
        Paragraph(
            "Decision-support report generated from the current "
            "dashboard analysis.",
            styles["subtitle"],
        ),
        Paragraph(f"<b>Generated:</b> {generated}", styles["body"]),
        Paragraph(
            f"<b>Reviews analyzed:</b> {_fmt_int(context['total_reviews'])}",
            styles["body"],
        ),
        Paragraph(
            f"<b>Dataset date range:</b> {date_range}",
            styles["body"],
        ),
        Paragraph(
            f"<b>Current analysis period:</b> {period}",
            styles["body"],
        ),
    ]


def _section_executive(context, styles):
    rows = [
        ["Total reviews", _fmt_int(context["total_reviews"])],
        ["Topics discovered", _fmt_int(context["valid_topics"])],
        [
            "Current sentiment",
            _fmt_number(context["latest_sentiment"]),
        ],
        [
            "Negative reviews (latest period)",
            _fmt_pct(
                context["latest_negative_percentage"],
                already_percent=True,
            ),
        ],
        [
            "Highest issue severity",
            f"{_fmt_number(context['highest_severity'], 1)}/100",
        ],
        ["Feedback change status", context["drift_status"]],
        [
            "Concept drift score",
            _fmt_number(context["latest_drift"]),
        ],
    ]

    return [
        Paragraph("1. Executive Summary", styles["heading"]),
        Paragraph(
            "KPIs match the Executive Intelligence Overview on the dashboard. "
            "No additional metrics are introduced here.",
            styles["muted"],
        ),
        _make_table(
            ["Metric", "Value"],
            rows,
            styles,
            col_widths=[3.4 * inch, 3.5 * inch],
        ),
    ]


def _section_attention(context, styles):
    flow = [Paragraph("2. What Needs Attention?", styles["heading"])]
    row = context["attention_row"]

    if row is None:
        flow.append(
            Paragraph(
                "No customer issue currently requires attention.",
                styles["body"],
            )
        )
        return flow

    topic = _display_topic(row.get("topic_label"))
    flow.append(
        Paragraph(
            f"Highest-severity issue from the existing recommendation result: "
            f"<b>{_escape(topic)}</b>.",
            styles["body"],
        )
    )

    detail_rows = [
        ["Business topic", topic],
        [
            "Severity score",
            f"{_fmt_number(row.get('severity_score'), 1)}/100",
        ],
        ["Severity level", _escape(row.get("severity_level"))],
        ["Frequency", _fmt_int(row.get("frequency"))],
        ["Negative ratio", _fmt_pct(row.get("negative_ratio"))],
        ["Growth rate", _fmt_pct(row.get("growth_rate"))],
        [
            "Concept drift score",
            _fmt_number(row.get("concept_drift_score")),
        ],
    ]

    flow.append(
        _make_table(
            ["Field", "Value"],
            detail_rows,
            styles,
            col_widths=[2.4 * inch, 4.5 * inch],
        )
    )
    flow.append(Spacer(1, 8))
    flow.append(
        Paragraph(
            f"<b>Recommended action:</b> "
            f"{_escape(row.get('recommendation'))}",
            styles["body"],
        )
    )
    return flow


def _section_sentiment(context, styles):
    flow = [Paragraph("3. Customer Sentiment", styles["heading"])]
    sentiment_df = context["sentiment_df"]

    if sentiment_df.empty:
        flow.append(
            Paragraph("No sentiment trend data is available.", styles["body"])
        )
        return flow

    flow.append(
        Paragraph(
            f"Latest-period average sentiment: "
            f"<b>{_fmt_number(context['latest_sentiment'])}</b>. "
            f"Negative reviews in the latest period: "
            f"<b>{_fmt_pct(context['latest_negative_percentage'], True)}</b>.",
            styles["body"],
        )
    )

    chart = _sentiment_chart(sentiment_df)
    if chart is not None:
        flow.append(Spacer(1, 6))
        flow.append(chart)

    if "negative_reviews" in sentiment_df.columns:
        neg_chart = _negative_volume_chart(sentiment_df)
        if neg_chart is not None:
            flow.append(Spacer(1, 8))
            flow.append(neg_chart)

    table_rows = []
    for _, row in sentiment_df.sort_values("month").iterrows():
        table_rows.append(
            [
                _escape(row.get("month")),
                _fmt_number(row.get("avg_sentiment")),
                _fmt_int(row.get("negative_reviews")),
                _fmt_int(row.get("total_reviews")),
            ]
        )

    flow.append(Spacer(1, 8))
    flow.append(
        _make_table(
            ["Month", "Avg sentiment", "Negative reviews", "Total reviews"],
            table_rows,
            styles,
            col_widths=[1.6 * inch, 1.7 * inch, 1.8 * inch, 1.8 * inch],
        )
    )
    return flow


def _sentiment_chart(sentiment_df):
    try:
        fig, ax = plt.subplots(figsize=(7.2, 2.6))
        ax.plot(
            sentiment_df["month"].astype(str),
            sentiment_df["avg_sentiment"],
            marker="o",
            color="#1f2937",
            linewidth=1.6,
        )
        ax.axhline(0, color="#9ca3af", linestyle="--", linewidth=0.8)
        ax.set_title("Customer Sentiment Over Time")
        ax.set_xlabel("Month")
        ax.set_ylabel("Average Sentiment")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        fig.tight_layout()
        return _chart_image(fig)
    except Exception:
        plt.close("all")
        return None


def _negative_volume_chart(sentiment_df):
    try:
        fig, ax = plt.subplots(figsize=(7.2, 2.6))
        ax.plot(
            sentiment_df["month"].astype(str),
            sentiment_df["negative_reviews"],
            marker="o",
            color="#374151",
            linewidth=1.6,
        )
        ax.set_title("Negative Review Volume")
        ax.set_xlabel("Month")
        ax.set_ylabel("Negative Reviews")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        fig.tight_layout()
        return _chart_image(fig)
    except Exception:
        plt.close("all")
        return None


def _section_prioritization(context, styles):
    flow = [
        Paragraph("4. Issue Prioritization", styles["heading"]),
        Paragraph(
            "Issues are ranked using the existing severity result "
            "(negative sentiment, frequency, growth, and feedback drift). "
            "No new severity calculation is performed.",
            styles["muted"],
        ),
    ]

    latest_severity_df = context["latest_severity_df"]
    if latest_severity_df.empty:
        flow.append(
            Paragraph(
                "No prioritized issues are available for the latest period.",
                styles["body"],
            )
        )
        return flow

    top = latest_severity_df.head(10)
    rec_lookup = {}
    recs = context["latest_recommendations"]
    if not recs.empty:
        rec_lookup = dict(
            zip(recs["topic_label"].astype(str), recs["recommendation"])
        )

    rows = []
    for _, row in top.iterrows():
        label = str(row.get("topic_label"))
        rows.append(
            [
                _display_topic(label),
                _fmt_number(row.get("severity_score"), 1),
                _escape(row.get("severity_level")),
                _fmt_int(row.get("frequency")),
                _fmt_pct(row.get("negative_ratio")),
                _fmt_pct(row.get("growth_rate")),
                _fmt_number(row.get("concept_drift_score")),
                rec_lookup.get(label, "N/A"),
            ]
        )

    flow.append(
        _make_table(
            [
                "Topic",
                "Severity",
                "Level",
                "Freq.",
                "Neg. ratio",
                "Growth",
                "Drift",
                "Recommendation",
            ],
            rows,
            styles,
            col_widths=[
                0.85 * inch,
                0.65 * inch,
                0.6 * inch,
                0.5 * inch,
                0.7 * inch,
                0.6 * inch,
                0.55 * inch,
                2.45 * inch,
            ],
        )
    )
    return flow


def _section_drift(context, styles):
    flow = [Paragraph("5. What Changed?", styles["heading"])]
    similarity_df = context["similarity_df"]

    if similarity_df.empty:
        flow.append(
            Paragraph(
                "Not enough historical data for drift analysis.",
                styles["body"],
            )
        )
        return flow

    latest = similarity_df.iloc[-1]
    flow.append(
        Paragraph(
            "Existing drift implementation is used without new thresholds. "
            f"Topic cosine threshold = {DRIFT_COSINE_THRESHOLD:.2f}. "
            f"Combined concept-drift alert threshold = "
            f"{CONCEPT_DRIFT_ALERT_THRESHOLD:.2f}. "
            "Dashboard status: Stable &lt; 0.35, Elevated 0.35–0.60, "
            "Critical ≥ 0.60.",
            styles["muted"],
        )
    )

    summary_rows = [
        ["Feedback change status", context["drift_status"]],
        [
            "Concept drift score",
            _fmt_number(latest.get("concept_drift_score")),
        ],
        [
            "Cosine similarity",
            _fmt_number(latest.get("cosine_similarity")),
        ],
        ["Topic change", _fmt_number(latest.get("topic_drift"))],
        ["Sentiment change", _fmt_number(latest.get("sentiment_drift"))],
        ["Volume change", _fmt_number(latest.get("volume_drift"))],
        [
            "Previous month",
            _escape(latest.get("previous_month")),
        ],
        [
            "Current month",
            _escape(latest.get("current_month")),
        ],
    ]
    flow.append(
        _make_table(
            ["Field", "Value"],
            summary_rows,
            styles,
            col_widths=[3.0 * inch, 3.9 * inch],
        )
    )

    chart = _drift_chart(similarity_df)
    if chart is not None:
        flow.append(Spacer(1, 8))
        flow.append(chart)

    table_rows = []
    for _, row in similarity_df.iterrows():
        table_rows.append(
            [
                _escape(row.get("previous_month")),
                _escape(row.get("current_month")),
                _fmt_number(row.get("cosine_similarity")),
                _fmt_number(row.get("concept_drift_score")),
                "Yes" if bool(row.get("concept_drift_detected")) else "No",
            ]
        )

    flow.append(Spacer(1, 8))
    flow.append(
        _make_table(
            [
                "Previous month",
                "Current month",
                "Cosine similarity",
                "Concept drift",
                "Alert",
            ],
            table_rows,
            styles,
            col_widths=[1.3 * inch, 1.3 * inch, 1.4 * inch, 1.3 * inch, 1.6 * inch],
        )
    )

    if context["alerts"]:
        flow.append(Spacer(1, 8))
        flow.append(Paragraph("<b>Detected drift alerts</b>", styles["body"]))
        for alert in context["alerts"]:
            flow.append(Paragraph(f"• {_escape(alert)}", styles["body"]))
    else:
        flow.append(
            Paragraph("No drift alerts were generated.", styles["body"])
        )

    return flow


def _drift_chart(similarity_df):
    try:
        fig, ax = plt.subplots(figsize=(7.2, 2.6))
        ax.plot(
            similarity_df["current_month"].astype(str),
            similarity_df["concept_drift_score"],
            marker="o",
            color="#1f2937",
            linewidth=1.6,
        )
        ax.axhline(
            CONCEPT_DRIFT_ALERT_THRESHOLD,
            color="#6b7280",
            linestyle="--",
            linewidth=0.8,
            label="Change threshold (0.35)",
        )
        ax.set_ylim(0, 1)
        ax.set_title("Overall Customer Feedback Change")
        ax.set_xlabel("Month")
        ax.set_ylabel("Concept Drift Score")
        ax.legend(fontsize=7)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        fig.tight_layout()
        return _chart_image(fig)
    except Exception:
        plt.close("all")
        return None


def _topic_latest_rows(latest_topics, latest_drift):
    work = latest_topics.copy()
    if "topic_label" in work.columns:
        grouped = work.groupby("topic_label", as_index=False)
        agg = {}
        if "frequency" in work.columns:
            agg["frequency"] = "sum"
        if "topic_share" in work.columns:
            agg["topic_share"] = "sum"
        if "topic_sentiment" in work.columns:
            agg["topic_sentiment"] = "mean"
        if "negative_ratio" in work.columns:
            agg["negative_ratio"] = "mean"
        if "growth_rate" in work.columns:
            agg["growth_rate"] = "mean"
        if agg:
            work = grouped.agg(agg)
            if "frequency" in work.columns:
                work = work.sort_values("frequency", ascending=False)

    rows = []
    for _, row in work.iterrows():
        rows.append(
            [
                _display_topic(row.get("topic_label")),
                _fmt_int(row.get("frequency")),
                _fmt_pct(row.get("topic_share")),
                _fmt_number(row.get("topic_sentiment")),
                _fmt_pct(row.get("negative_ratio")),
                _fmt_pct(row.get("growth_rate")),
                _fmt_number(latest_drift),
            ]
        )
    return rows


def _section_topics(context, styles):
    flow = [Paragraph("6. Topic Intelligence", styles["heading"])]
    topic_frequency_df = context["topic_frequency_df"]
    latest_month = context["latest_month"]

    if topic_frequency_df.empty:
        flow.append(
            Paragraph(
                "No usable topic information is available.",
                styles["body"],
            )
        )
        return flow

    latest_topics = topic_frequency_df.copy()
    if latest_month is not None and "month" in latest_topics.columns:
        latest_topics = latest_topics[
            latest_topics["month"] == latest_month
        ]

    if "topic_label" in latest_topics.columns:
        latest_topics = latest_topics[
            latest_topics["topic_label"] != "outlier"
        ]

    if latest_topics.empty:
        flow.append(
            Paragraph(
                "No business topics were available for the latest period.",
                styles["body"],
            )
        )
        return flow

    rows = _topic_latest_rows(latest_topics, context["latest_drift"])

    flow.append(
        Paragraph(
            f"Topic statistics for the latest analysis period "
            f"({latest_month}). Concept drift is the existing monthly "
            f"score for that period; the pipeline does not compute a "
            f"separate per-topic drift value.",
            styles["muted"],
        )
    )
    flow.append(
        _make_table(
            [
                "Topic",
                "Frequency",
                "Share",
                "Sentiment",
                "Neg. ratio",
                "Growth",
                "Drift",
            ],
            rows,
            styles,
            col_widths=[
                1.3 * inch,
                0.9 * inch,
                0.8 * inch,
                0.9 * inch,
                0.9 * inch,
                0.8 * inch,
                1.3 * inch,
            ],
        )
    )

    if "month" in topic_frequency_df.columns:
        history_rows = []
        history = (
            topic_frequency_df[
                topic_frequency_df["topic_label"] != "outlier"
            ]
            .sort_values(["topic_label", "month"])
            .head(80)
        )
        for _, row in history.iterrows():
            history_rows.append(
                [
                    _escape(row.get("month")),
                    _display_topic(row.get("topic_label")),
                    _fmt_int(row.get("frequency")),
                    _fmt_number(row.get("topic_sentiment")),
                    _fmt_pct(row.get("growth_rate")),
                ]
            )
        if history_rows:
            flow.append(Spacer(1, 10))
            flow.append(
                Paragraph(
                    "Topic time series already produced by temporal "
                    "aggregation (truncated if long):",
                    styles["muted"],
                )
            )
            flow.append(
                _make_table(
                    [
                        "Month",
                        "Topic",
                        "Frequency",
                        "Sentiment",
                        "Growth",
                    ],
                    history_rows,
                    styles,
                    col_widths=[
                        1.2 * inch,
                        1.8 * inch,
                        1.2 * inch,
                        1.3 * inch,
                        1.4 * inch,
                    ],
                )
            )

    return flow


def _section_sku(context, styles):
    sku_summary = context["sku_summary"]
    if sku_summary.empty:
        return []

    flow = [
        Paragraph("7. Product / SKU Intelligence", styles["heading"]),
        Paragraph(
            "Included because the uploaded dataset contains a sku column. "
            "Values use the same SKU aggregation as the dashboard.",
            styles["muted"],
        ),
    ]

    display = sku_summary.copy()
    rows = []
    for _, row in display.iterrows():
        rows.append(
            [
                _escape(row.get("sku")),
                _fmt_int(row.get("total_reviews")),
                _fmt_number(row.get("avg_sentiment"), 3),
                _fmt_pct(
                    row.get("negative_percentage"),
                    already_percent=True,
                ),
            ]
        )

    flow.append(
        _make_table(
            ["SKU", "Review volume", "Average sentiment", "Negative %"],
            rows,
            styles,
            col_widths=[1.8 * inch, 1.6 * inch, 1.8 * inch, 1.7 * inch],
        )
    )
    return flow


def _section_recommendations(context, styles):
    heading_number = "8" if not context["sku_summary"].empty else "7"
    flow = [
        Paragraph(
            f"{heading_number}. Recommended Actions",
            styles["heading"],
        ),
        Paragraph(
            "Recommendations come from the existing rule-based "
            "RecommendationEngine. They are not generated by an LLM.",
            styles["muted"],
        ),
    ]

    recs = context["latest_recommendations"]
    if recs.empty:
        flow.append(
            Paragraph(
                "No corrective recommendations are available.",
                styles["body"],
            )
        )
        return flow

    for _, row in recs.iterrows():
        topic = _display_topic(row.get("topic_label"))
        block = [
            Paragraph(
                f"<b>{_escape(topic)}</b> — "
                f"{_fmt_number(row.get('severity_score'), 1)}/100 — "
                f"{_escape(row.get('severity_level'))}",
                styles["body"],
            ),
            Paragraph(_escape(row.get("recommendation")), styles["body"]),
        ]
        flow.append(KeepTogether(block))

    return flow


def _section_insights(context, styles):
    heading_number = "9" if not context["sku_summary"].empty else "8"
    flow = [
        Paragraph(
            f"{heading_number}. Explainable Insights",
            styles["heading"],
        ),
        Paragraph(
            "Text is produced by the existing InsightExplainer.",
            styles["muted"],
        ),
    ]

    if context["sentiment_insight"]:
        flow.append(
            Paragraph(
                f"<b>Sentiment:</b> {_escape(context['sentiment_insight'])}",
                styles["body"],
            )
        )
    if context["topic_insight"]:
        flow.append(
            Paragraph(
                f"<b>Topic:</b> {_escape(context['topic_insight'])}",
                styles["body"],
            )
        )
    if context["drift_insight"]:
        flow.append(
            Paragraph(
                f"<b>Change:</b> {_escape(context['drift_insight'])}",
                styles["body"],
            )
        )

    if (
        not context["sentiment_insight"]
        and not context["topic_insight"]
        and not context["drift_insight"]
    ):
        flow.append(
            Paragraph(
                "Explainable insights were not available for this run.",
                styles["body"],
            )
        )

    return flow


def _section_methodology(styles):
    heading_number_note = (
        "This report is generated from already-computed pipeline "
        "results. The download action does not rerun preprocessing, "
        "VADER, BERTopic, drift detection, or severity scoring."
    )
    return [
        Paragraph("Methodology Note", styles["heading"]),
        Paragraph(heading_number_note, styles["body"]),
        Paragraph(
            "Topics are discovered using BERTopic. Sentiment uses VADER "
            "compound scores. Topic drift uses cosine similarity on monthly "
            "topic distributions. Combined concept drift also incorporates "
            "normalized sentiment and volume change using the existing "
            "pipeline weights and thresholds. Severity is a relative 0–100 "
            "decision-support score. Recommendations are deterministic and "
            "rule-based. Results are intended for decision support and are "
            "not legal, financial, SLA, or operational risk scores. This "
            "system has not been presented here as a production-validated "
            "accuracy benchmark.",
            styles["body"],
        ),
    ]
