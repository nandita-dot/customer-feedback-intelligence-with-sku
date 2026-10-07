from io import BytesIO
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.reporting._smoke_report import _sample_result


def _csv_bytes(include_sku=True):
    df = pd.DataFrame(
        {
            "review": ["Delivery was late", "Price is high"],
            "date": ["2025-11-01", "2025-12-01"],
        }
    )
    if include_sku:
        df["sku"] = ["SKU-1", "SKU-2"]
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()


def test_empty_state_has_no_download():
    at = AppTest.from_file("src/dashboard/app.py", default_timeout=180)
    at.run()
    assert any(
        "Select a dataset above to begin." in (el.value or "")
        for el in at.info
    )
    assert len(at.get("download_button")) == 0


def _select_custom_upload(at, filename, csv_bytes):
    at.selectbox[0].select("Custom CSV").run()
    at.get("file_uploader")[0].upload(
        filename,
        csv_bytes,
        "text/csv",
    ).run()
    at.button[0].click().run()


def test_download_does_not_rerun_pipeline():
    calls = {"n": 0}

    def fake_pipeline(df):
        calls["n"] += 1
        return _sample_result(include_sku="sku" in df.columns)

    csv_bytes = _csv_bytes(include_sku=True)

    with patch("src.run_pipeline.run_pipeline", side_effect=fake_pipeline):
        at = AppTest.from_file("src/dashboard/app.py", default_timeout=180)
        at.run()
        _select_custom_upload(at, "upload.csv", csv_bytes)
        first_calls = calls["n"]
        assert first_calls == 1
        assert len(at.get("download_button")) == 1

        # Simulate a Streamlit rerun, as download_button causes.
        at.run()
        assert calls["n"] == 1
        assert len(at.get("download_button")) == 1


def test_without_sku_still_builds_report():
    calls = {"n": 0}

    def fake_pipeline(df):
        calls["n"] += 1
        return _sample_result(include_sku=False)

    csv_bytes = _csv_bytes(include_sku=False)

    with patch("src.run_pipeline.run_pipeline", side_effect=fake_pipeline):
        at = AppTest.from_file("src/dashboard/app.py", default_timeout=180)
        at.run()
        _select_custom_upload(at, "upload_no_sku.csv", csv_bytes)
        assert calls["n"] == 1
        assert len(at.get("download_button")) == 1


if __name__ == "__main__":
    test_empty_state_has_no_download()
    print("empty ok")
    test_download_does_not_rerun_pipeline()
    print("cache ok")
    test_without_sku_still_builds_report()
    print("no sku ok")
