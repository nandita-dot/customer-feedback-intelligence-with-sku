from typing import List, Dict

from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel

print("LOADING LDA TOPIC MODEL")

class LDATopicModel:
    """
    LDA Topic Modeling Pipeline using gensim.
    """

    def __init__(
        self,
        num_topics: int = 5,
        passes: int = 10,
        random_state: int = 42
    ):

        self.num_topics = num_topics
        self.passes = passes
        self.random_state = random_state

        self.dictionary = None
        self.corpus = None
        self.lda_model = None

    def build_dictionary(
        self,
        tokenized_reviews: List[List[str]]
    ):
        """
        Create gensim dictionary.
        """

        self.dictionary = corpora.Dictionary(
            tokenized_reviews
        )

        # Remove extreme words
        self.dictionary.filter_extremes(
            no_below=2,
            no_above=0.8
        )

    def vectorize_reviews(
        self,
        tokenized_reviews: List[List[str]]
    ):
        """
        Convert tokenized reviews into BoW vectors.
        """

        self.corpus = [
            self.dictionary.doc2bow(review)
            for review in tokenized_reviews
        ]

    def train_model(self):
        """
        Train LDA model.
        """

        self.lda_model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            passes=self.passes,
            random_state=self.random_state
        )

    def compute_coherence(
        self,
        tokenized_reviews: List[List[str]]
    ) -> float:
        """
        Compute topic coherence score.
        """

        coherence_model = CoherenceModel(
            model=self.lda_model,
            texts=tokenized_reviews,
            dictionary=self.dictionary,
            coherence="c_v"
        )

        coherence_score = coherence_model.get_coherence()

        return coherence_score

    def extract_topics(
        self,
        num_words: int = 10
    ) -> List[Dict]:
        """
        Extract discovered topics.
        """

        topics = []

        for topic_id, topic_words in self.lda_model.print_topics(
            num_words=num_words
        ):

            topics.append({
                "topic_id": topic_id,
                "words": topic_words
            })

        return topics

    def fit(
        self,
        tokenized_reviews: List[List[str]]
    ):
        """
        Complete training pipeline.
        """

        self.build_dictionary(tokenized_reviews)

        self.vectorize_reviews(tokenized_reviews)

        self.train_model()

    def get_document_topics(
        self,
        tokenized_review: List[str]
    ):
        """
        Get topic probabilities for a review.
        """

        bow_vector = self.dictionary.doc2bow(
            tokenized_review
        )

        return self.lda_model.get_document_topics(
            bow_vector
        )
    
print(LDATopicModel)