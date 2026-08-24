from typing import List
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel

class LDATopicModel:

    def __init__(
        self,
        min_topics=2,
        max_topics=10,
        passes=10,
        random_state=42
    ):

        self.min_topics = min_topics
        self.max_topics = max_topics
        self.passes = passes
        self.random_state = random_state

        self.dictionary = None
        self.corpus = None
        self.model = None
        self.best_topic_count = None

    def fit(self, tokenized_reviews: List[List[str]]):
        self.dictionary = corpora.Dictionary(tokenized_reviews)
        self.dictionary.filter_extremes(
            no_below=2,
            no_above=0.8
        )

        self.corpus = [self.dictionary.doc2bow(text)
            for text in tokenized_reviews
        ]

        best_score = -1
        best_model = None

        for k in range(self.min_topics, self.max_topics + 1):
            lda = LdaModel(
                corpus=self.corpus,
                id2word=self.dictionary,
                num_topics=k,
                passes=self.passes,
                random_state=self.random_state
            )

            coherence_model = CoherenceModel(
                model=lda,
                texts=tokenized_reviews,
                dictionary=self.dictionary,
                coherence="c_v"
            )

            score = coherence_model.get_coherence()
            print(f"Topics={k} | Coherence={score:.4f}")

            if score > best_score:
                best_score = score
                best_model = lda
                self.best_topic_count = k

        self.model = best_model

        print(f"\nBest topic count: {self.best_topic_count}")
        print(f"Best coherence score: {best_score:.4f}")

        def extract_topics(self, num_words=10):
            topics = []
            for topic_id, words in self.model.print_topics(num_words=num_words):

                topics.append({
                    "topic_id": topic_id,
                    "words": words
                })

            return topics
        
        def get_document_topics(self, tokens):
            bow = self.dictionary.doc2bow(tokens)
            return self.model.get_document_topics(bow)