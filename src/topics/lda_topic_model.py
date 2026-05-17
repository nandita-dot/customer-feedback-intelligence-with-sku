from typing import List, Dict
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel


class LDATopicModel:

    def __init__(self, num_topics=5, passes=10, random_state=42):
        self.num_topics = num_topics
        self.passes = passes
        self.random_state = random_state

        self.dictionary = None
        self.corpus = None
        self.model = None

    def fit(self, texts: List[List[str]]):

        self.dictionary = corpora.Dictionary(texts)
        self.dictionary.filter_extremes(no_below=2, no_above=0.8)

        self.corpus = [self.dictionary.doc2bow(t) for t in texts]

        self.model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            passes=self.passes,
            random_state=self.random_state
        )

    def get_document_topics(self, tokens: List[str]):
        bow = self.dictionary.doc2bow(tokens)
        return self.model.get_document_topics(bow)

    def extract_topics(self):
        return [
            {
                "topic_id": i,
                "words": w
            }
            for i, w in self.model.print_topics()
        ]