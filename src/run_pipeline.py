from nltk import download

from src.preprocessing.data_loader import ReviewDataLoader
from src.preprocessing.text_preprocessor import TextPreprocessor
from src.sentiment.sentiment_analyzer import SentimentAnalyzer
from src.topics.lda_topic_model import LDATopicModel
from src.temporal.temporal_aggregator import TemporalAggregator


def run_pipeline(file_path: str):

    # ----------------------------
    # NLTK setup
    # ----------------------------
    download("stopwords")

    # ----------------------------
    # LOAD DATA
    # ----------------------------
    loader = ReviewDataLoader(file_path)
    df = loader.load_data()

    # ----------------------------
    # TEXT PREPROCESSING
    # ----------------------------
    preprocessor = TextPreprocessor()

    tokens_list = []
    cleaned_texts = []

    for review in df["review"]:
        tokens, clean = preprocessor.preprocess(review)
        tokens_list.append(tokens)
        cleaned_texts.append(clean)

    df["cleaned_review"] = cleaned_texts

    # ----------------------------
    # SENTIMENT ANALYSIS
    # ----------------------------
    sentiment = SentimentAnalyzer()

    df["compound_score"] = df["cleaned_review"].apply(
        lambda x: sentiment.analyze_review(x)["compound_score"]
    )

    # ----------------------------
    # TOPIC MODELING
    # ----------------------------
    lda = LDATopicModel(num_topics=5)
    lda.fit(tokens_list)

    df["topic_id"] = [
        max(lda.get_document_topics(t), key=lambda x: x[1])[0]
        for t in tokens_list
    ]

    # ----------------------------
    # TEMPORAL ANALYSIS
    # ----------------------------
    aggregator = TemporalAggregator()
    results = aggregator.aggregate_monthly(df)

    return df, results