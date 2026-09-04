# Customer Feedback Intelligence System

> An AI-powered customer feedback analytics platform that transforms unstructured customer reviews into actionable, explainable business intelligence.

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![BERTopic](https://img.shields.io/badge/NLP-BERTopic-orange)](https://maartengr.github.io/BERTopic/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Dashboard Preview

![Customer Feedback Intelligence Dashboard](assets/dashboard-preview.png)

> **Tip:** Replace `assets/dashboard-preview.png` with a screenshot of your actual Streamlit dashboard. A real product screenshot is strongly recommended over a generic AI image.

---

## Overview

The **Customer Feedback Intelligence System** is an NLP-driven business intelligence application designed to analyze large volumes of unstructured customer feedback.

Instead of simply displaying sentiment scores or review counts, the system attempts to answer four important business questions:

> **What are customers talking about?**
> **What is changing over time?**
> **Which issues require the most attention?**
> **What corrective action should be taken?**

The system combines:

* Sentiment analysis
* Automatic topic discovery
* Business-friendly topic labeling
* Temporal topic analysis
* Feedback drift detection
* Issue severity scoring
* Issue prioritization
* Product/SKU intelligence
* Corrective-action recommendations
* Explainable business insights
* Interactive Streamlit visualization

The application follows a modular architecture so that individual analytical components can be improved without unnecessarily changing the entire pipeline.

---

## Problem Statement

Customer reviews contain valuable information about product quality, pricing, delivery, customer support, and overall customer experience. However, manually analyzing large volumes of feedback makes it difficult to identify emerging issues and determine which problems deserve immediate attention.

Many basic feedback-analysis systems stop at sentiment analysis or review visualization.

This project extends that workflow by combining:

```text
Topic Discovery
      +
Sentiment Analysis
      +
Temporal Analysis
      +
Feedback Drift Detection
      +
Severity Scoring
      +
Product Intelligence
      +
Corrective Recommendations
```

to create a unified customer feedback intelligence pipeline.

The objective is to move from simply **reporting what happened** to identifying **what is changing, what matters most, and what action should be considered**.

---

## How It Works

```text
Customer Reviews
       ↓
Data Validation
       ↓
Text Preprocessing
       ↓
Sentiment + Topic Analysis
       ↓
Temporal Analysis
       ↓
Drift Detection
       ↓
Severity & Issue Prioritization
       ↓
Corrective Recommendations
       ↓
Explainable Business Insights
       ↓
Interactive Dashboard
```

---

## Key Features

### 🧠 Sentiment Analysis

Uses **VADER Sentiment Analysis** to classify customer feedback as:

* Positive
* Neutral
* Negative

Each review receives a compound sentiment score between approximately `-1` and `+1`.

Current thresholds:

| Compound Score | Sentiment |
| -------------- | --------- |
| `>= 0.05`      | Positive  |
| `<= -0.05`     | Negative  |
| Otherwise      | Neutral   |

Sentiment scores are also used by downstream severity and business-insight components.

---

### 🔎 Automatic Topic Discovery

Uses **BERTopic** to discover recurring themes in customer feedback.

For example:

```text
"The package arrived two weeks late."

"Courier delivery was extremely slow."

"My order is still being shipped."
```

may be grouped into a broader:

```text
Delivery
```

topic.

BERTopic initially produces numerical topic IDs. These are then converted into business-facing labels using a custom topic-labeling layer.

Current business-facing topic examples include:

* Delivery
* Pricing
* Support
* Quality
* Quantity
* General Feedback

---

### 📅 Temporal Topic Analysis

Customer feedback is analyzed across monthly time periods to identify changes in:

* Review volume
* Sentiment
* Topic activity
* Topic growth
* Customer discussion trends

The system also uses BERTopic's `topics_over_time` functionality for temporal topic analysis.

This allows the system to identify patterns such as:

```text
Delivery complaints

April      18
May        24
June       41
July       78
```

indicating that delivery has become an increasingly important customer concern.

---

### 🚨 Feedback Drift Detection

The system monitors changes in customer feedback patterns over time.

Drift/change analysis considers:

* Topic distribution changes
* Sentiment changes
* Review-volume changes

Topic drift uses cosine similarity/distance, while sentiment and volume changes are normalized before contributing to the overall feedback-change analysis.

The dashboard can classify the current state, such as:

```text
Stable
```

and can expose detected drift alerts when significant changes occur.

---

### 📊 Issue Severity Scoring

Customer issues receive a severity score on a:

```text
0 – 100
```

scale.

Severity incorporates multiple signals including:

* Negative sentiment
* Issue frequency
* Topic growth
* Overall feedback drift

Issues can then be categorized as:

```text
Low
Medium
High
Critical
```

This enables the system to prioritize issues instead of treating every customer complaint equally.

---

### 🎯 Issue Prioritization

Customer issues are ranked using multiple signals:

```text
Negative Sentiment
        +
Frequency
        +
Growth
        +
Feedback Drift
        ↓
Issue Priority
```

This helps identify issues that may require more immediate business attention.

---

### 🛍️ Product / SKU Intelligence

When SKU information is available, the system analyzes feedback at the product level.

This can help identify:

* Products receiving unusually negative feedback
* Product-specific quality issues
* Delivery problems affecting particular products
* SKU-level sentiment patterns
* Products requiring further investigation

---

### 💡 Corrective Recommendations

The system converts high-priority customer issues into potential corrective actions.

#### Delivery

```text
Investigate logistics and delivery partners.
Review delayed shipments and improve tracking visibility.
```

#### Pricing

```text
Review pricing competitiveness, discounts
and customer value perception.
```

#### Support

```text
Review customer support response times
and increase support capacity for recurring complaints.
```

#### Quality

```text
Investigate product quality and manufacturing issues.
Perform quality-control checks on affected products.
```

Recommendations are generated from the detected customer issue and its analytical context.

---

### 🔍 Explainable Business Insights

The system translates analytical results into human-readable business statements.

Instead of presenting only:

```text
Average Sentiment = -0.42
```

the system can communicate the result in a business-oriented form such as:

```text
Customer sentiment is currently negative,
indicating increasing customer dissatisfaction.
```

The explainability layer is designed to make analytical results understandable to users who do not need to inspect the underlying calculations.

---

# System Architecture

```text
                         CUSTOMER REVIEWS
                                │
                                ▼
                         DATA VALIDATION
                                │
                                ▼
                       TEXT PREPROCESSING
                                │
                                ▼
                     SENTIMENT ANALYSIS
                         (VADER)
                                │
                                ▼
                      TOPIC DISCOVERY
                         (BERTopic)
                                │
                                ▼
                       TOPIC LABELING
                                │
                                ▼
                    TEMPORAL AGGREGATION
                                │
                ┌───────────────┼───────────────┐
                │               │               │
                ▼               ▼               ▼
          Topic Trends    Sentiment Trends   SKU Analysis
                │               │               │
                └───────────────┼───────────────┘
                                │
                                ▼
                       DRIFT DETECTION
                                │
                                ▼
                       SEVERITY SCORING
                                │
                                ▼
                      ISSUE PRIORITIZATION
                                │
                                ▼
                    RECOMMENDATION ENGINE
                                │
                                ▼
                       EXPLAINABILITY
                                │
                                ▼
                    STREAMLIT DASHBOARD
```

---

# Project Objectives

The project is built around five primary objectives.

### Objective 1 — Temporal Topic Modelling

To implement temporal topic modelling for analysing changing customer discussion trends over time.

### Objective 2 — Concept Drift Detection

To develop concept drift detection mechanisms for identifying sudden changes in customer sentiment and feedback patterns.

### Objective 3 — Severity Impact Scoring

To design a severity impact scoring mechanism to prioritize critical customer issues.

### Objective 4 — Corrective Recommendation Module

To build a recommendation module that provides corrective suggestions based on detected feedback patterns.

### Objective 5 — Interactive Dashboard

To develop an interactive dashboard for visualizing topics, sentiment trends, and drift analysis results.

---

# Technology Stack

| Category              | Technology                                           |
| --------------------- | ---------------------------------------------------- |
| Programming Language  | Python                                               |
| Dashboard             | Streamlit                                            |
| Data Processing       | Pandas                                               |
| Sentiment Analysis    | VADER Sentiment                                      |
| Topic Modelling       | BERTopic                                             |
| NLP / Text Processing | spaCy, NLTK                                          |
| Machine Learning      | scikit-learn                                         |
| Temporal Analysis     | Pandas, BERTopic `topics_over_time`                  |
| Drift Detection       | Cosine Similarity, normalized sentiment/volume drift |
| Visualization         | Plotly                                               |
| Version Control       | Git                                                  |

The authoritative dependency list is maintained in [`requirements.txt`](requirements.txt).

---

# Input Dataset

The application accepts customer feedback in CSV format.

## Required Columns

```text
review
date
```

## Optional Column

```text
sku
```

### Column Description

| Column   | Description            | Required |
| -------- | ---------------------- | -------- |
| `review` | Customer feedback text | Yes      |
| `date`   | Date of the feedback   | Yes      |
| `sku`    | Product/SKU identifier | No       |

### Example

```csv
review,date,sku
"The product quality is excellent",2026-01-05,SKU-101
"Delivery was extremely late",2026-01-08,SKU-102
"Too expensive for the quality",2026-01-12,SKU-103
"Customer support solved my issue quickly",2026-01-17,SKU-101
```

> **Note:** Development and testing use a realistic synthetic/dummy dataset. It is not real customer data and should not be represented as such.

---

# Data Processing Pipeline

## 1. Data Validation

The uploaded dataset is validated before entering the NLP pipeline.

```text
CSV
 ↓
Column Validation
 ↓
Date Validation
 ↓
Data Cleaning
```

---

## 2. Text Preprocessing

Customer reviews undergo preprocessing before topic modelling.

The process includes:

```text
Lowercasing
     ↓
URL Removal
     ↓
Number Removal
     ↓
Punctuation Removal
     ↓
Whitespace Normalization
     ↓
Stopword Removal
     ↓
Lemmatization
```

---

## 3. Sentiment Analysis

VADER analyzes each review and produces:

```text
compound_score
sentiment_label
```

These outputs are later used by:

* Temporal analysis
* Severity scoring
* Issue prioritization
* Business insights

---

## 4. Topic Modelling

BERTopic discovers semantic groups within customer reviews.

The current model configuration includes:

```text
min_topic_size = 2
nr_topics = None
```

The system also uses:

```text
topics_over_time
```

for temporal topic analysis.

---

## 5. Topic Labeling

Raw BERTopic topic IDs are converted into business-friendly labels using a custom keyword-overlap `TopicLabeler`.

The conceptual flow is:

```text
BERTopic Topic ID
        ↓
Top Topic Keywords
        ↓
Keyword Matching
        ↓
Business Topic Label
```

---

## 6. Temporal Aggregation

Customer feedback is aggregated into monthly analysis periods.

The system calculates:

* Review volume
* Average sentiment
* Topic frequency
* Topic activity
* Topic trends

---

## 7. Drift Detection

Changes across time are analyzed using multiple signals:

```text
Topic Distribution Change
          +
Sentiment Change
          +
Volume Change
          ↓
Overall Feedback Change
```

This enables the system to detect periods where customer feedback behavior changes significantly.

---

## 8. Severity Scoring

Detected issues receive a score between:

```text
0 – 100
```

using multiple analytical signals.

The resulting severity level can be:

```text
Low
Medium
High
Critical
```

---

## 9. Issue Prioritization

Issues are ranked according to their analytical impact.

```text
Detected Issue
      ↓
Frequency
      +
Negative Sentiment
      +
Growth
      +
Feedback Drift
      ↓
Priority
```

---

## 10. Recommendations

High-priority issues are passed to the recommendation engine.

```text
Detected Issue
      ↓
Topic
      ↓
Severity
      ↓
Feedback Pattern
      ↓
Corrective Recommendation
```

---

## 11. Explainability

The final analytical outputs are converted into natural-language business insights.

---

# Dashboard

The application is implemented using **Streamlit**.

## Executive Intelligence Overview

Provides high-level metrics such as:

* Current analysis period
* Total reviews
* Topics discovered
* Current sentiment
* Negative reviews
* Highest issue severity
* Critical issue status

---

## What Needs Attention?

Highlights the highest-priority customer issue.

Information may include:

* Topic
* Severity score
* Severity level
* Latest-period review count
* Negative review ratio
* Growth rate
* Reason for attention
* Recommended action

---

## Customer Sentiment

Visualizes how customer sentiment changes over time.

---

## Customer Issue Prioritization

Ranks customer issues using:

```text
Negative Sentiment
+
Frequency
+
Growth
+
Feedback Drift
```

---

## What Changed in Customer Feedback?

Displays changes in:

* Overall feedback
* Topics
* Sentiment
* Review volume

It can also expose detected drift alerts.

---

## Topic Intelligence

Allows users to investigate individual customer topics instead of displaying every topic simultaneously.

---

## Product / SKU Intelligence

Analyzes products receiving unusually negative feedback when SKU information is available.

---

## Recommended Corrective Actions

Displays corrective actions generated from high-priority customer issues.

---

## Explainable Business Insights

Provides natural-language interpretations of analytical results.

---

## Discovered Topic Details

Provides additional information about the topics discovered by BERTopic.

---

## Processed Customer Feedback

Displays processed customer feedback generated by the NLP pipeline.

---

# Project Structure

The project follows a modular architecture.

```text
project-root/
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
│
├── assets/
│   └── dashboard-preview.png
│
├── src/
│   ├── preprocessing/
│   │   └── text_preprocessor.py
│   │
│   ├── sentiment/
│   │   └── sentiment_analyzer.py
│   │
│   ├── topics/
│   │   ├── bertopic_model.py
│   │   └── topic_labeler.py
│   │
│   ├── temporal/
│   │   └── temporal_aggregator.py
│   │
│   ├── drift/
│   │   └── topic_drift_detector.py
│   │
│   ├── severity/
│   │   └── ...
│   │
│   ├── recommendations/
│   │   └── ...
│   │
│   ├── explainability/
│   │   └── explainer.py
│   │
│   └── run_pipeline.py
│
└── data/
    └── dummy customer feedback dataset
```

> The exact file structure may evolve as the project develops.

---

# Installation

## 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd customer-feedback-intelligence
```

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

## 3. Activate the Environment

### Windows

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Usage

Start the Streamlit application from the project root:

```bash
streamlit run app.py
```

Then upload a CSV containing at least:

```text
review
date
```

and optionally:

```text
sku
```

The application will process the dataset and populate the intelligence dashboard.

---

# Example Workflow

```text
Upload Customer Feedback CSV
             ↓
       Preprocess Reviews
             ↓
      Analyze Sentiment
             ↓
       Discover Topics
             ↓
      Label Customer Issues
             ↓
     Analyze Temporal Trends
             ↓
       Detect Feedback Drift
             ↓
       Score Issue Severity
             ↓
       Prioritize Issues
             ↓
 Generate Corrective Recommendations
             ↓
       Display Insights
             ↓
      Streamlit Dashboard
```

---

# Example Business Scenario

Suppose customer feedback shows:

```text
Delivery complaints:

April      18 reviews
May        24 reviews
June       41 reviews
July       78 reviews
```

while sentiment changes from:

```text
April      -0.12
May        -0.19
June       -0.38
July       -0.61
```

The system can identify:

```text
Issue:
Delivery

Trend:
Rapidly Increasing

Sentiment:
Strongly Negative

Severity:
High / Critical
```

and generate a corrective action such as:

```text
Investigate logistics and delivery partners.
Review delayed shipments and improve tracking visibility.
```

This demonstrates the intended progression:

```text
Raw Reviews
    ↓
Detection
    ↓
Prioritization
    ↓
Explanation
    ↓
Action
```

---

# Research / Project Contribution

The project integrates multiple analytical signals into a unified customer feedback intelligence framework:

```text
Temporal Topic Modelling
        +
Sentiment Dynamics
        +
Concept Drift Detection
        +
Severity Impact Scoring
        +
SKU Impact Analysis
        +
Corrective Recommendations
        +
Explainable Visualization
```

The central contribution is an attempt to move beyond conventional feedback dashboards.

### Traditional Approach

```text
Reviews
   ↓
Sentiment
   ↓
Visualization
```

### Proposed Approach

```text
Reviews
   ↓
Sentiment + Topics
   ↓
Temporal Analysis
   ↓
Detect Change
   ↓
Measure Impact
   ↓
Prioritize Issues
   ↓
Recommend Action
   ↓
Explain Results
```

The system therefore aims to transform raw customer feedback into **prioritized, explainable, actionable intelligence**.

---

# Evaluation

The system can be evaluated across multiple analytical components.

## Sentiment Analysis

Potential metrics:

* Accuracy
* Precision
* Recall
* F1-score

---

## Topic Modelling

Potential evaluation criteria:

* Topic coherence
* Topic diversity
* Human interpretability

---

## Drift Detection

Potential evaluation criteria:

* True drift detection
* False positives
* False negatives
* Detection delay

---

## Severity Scoring

Severity rankings can be compared against manually ranked issues or domain-expert assessments.

---

## Recommendation Quality

Recommendations can be evaluated for:

* Relevance
* Correctness
* Actionability
* Consistency

---

# Limitations

The current system has several limitations:

* Development and testing currently rely on synthetic customer feedback.
* VADER is a general-purpose sentiment model and may not capture every domain-specific sentiment nuance.
* BERTopic topic quality depends on the quantity and quality of input reviews.
* Business topic labels depend on the current keyword-based labeling strategy.
* Drift thresholds may require calibration for different datasets.
* Severity scores are analytical prioritization signals and should not be interpreted as absolute measures of business risk.
* Recommendations are intended as decision-support suggestions rather than fully autonomous business decisions.

---

# Future Scope

Potential future improvements include:

### Advanced Topic Discovery

Improve topic coherence, topic naming, and semantic merging of similar topics.

### Advanced Sentiment Models

Evaluate transformer-based sentiment models against VADER for domain-specific customer feedback.

### Real-Time Feedback Ingestion

Integrate the system with:

* E-commerce platforms
* Customer-support systems
* CRM systems
* Survey platforms
* Review platforms

### Advanced Alerting

Trigger alerts when:

* Severity crosses a threshold
* Negative sentiment rises sharply
* A topic grows unusually quickly
* A new issue emerges
* Significant feedback drift occurs

### Advanced Product Intelligence

Expand SKU-level analysis for:

* Quality problems
* Delivery issues
* Product complaints
* Sentiment changes
* Emerging product-specific issues

### Model Evaluation

Add formal evaluation for:

* Sentiment classification
* Topic coherence
* Topic-label quality
* Drift detection
* Severity scoring
* Recommendation quality

---

# Development Status

## Implemented

* [x] CSV data ingestion
* [x] Data validation
* [x] Text preprocessing
* [x] VADER sentiment analysis
* [x] BERTopic topic modelling
* [x] Business topic labeling
* [x] Temporal aggregation
* [x] Temporal topic analysis
* [x] Topic drift detection
* [x] Sentiment drift analysis
* [x] Volume drift analysis
* [x] Severity scoring
* [x] Issue prioritization
* [x] Corrective recommendations
* [x] Explainable business insights
* [x] SKU intelligence
* [x] Interactive Streamlit dashboard

## Future Improvements

* [ ] Advanced topic merging
* [ ] Improved drift calibration
* [ ] Formal model evaluation
* [ ] Real-world dataset evaluation
* [ ] Advanced alerting
* [ ] Real-time data ingestion

---

# Links

* 💻 **Source Code:** [GitHub Repository](https://github.com/nandita-dot/customer-feedback-intelligence-with-sku)
* 📄 **Project Report:** (to be added soon)

---

# License

This project is licensed under the terms specified in the [`LICENSE`](LICENSE) file.

---

# Author

**Nandita Nair**

B.E. Computer Science
Acharya Institute of Technology

---

# Acknowledgements

This project makes use of the following open-source technologies:

* [BERTopic](https://maartengr.github.io/BERTopic/)
* [VADER Sentiment](https://github.com/cjhutto/vaderSentiment)
* [spaCy](https://spacy.io/)
* [NLTK](https://www.nltk.org/)
* [Pandas](https://pandas.pydata.org/)
* [scikit-learn](https://scikit-learn.org/)
* [Streamlit](https://streamlit.io/)
* [Plotly](https://plotly.com/)

---

## ⭐ Support

If you find this project interesting, consider giving the repository a ⭐.

Feedback and suggestions are welcome.
