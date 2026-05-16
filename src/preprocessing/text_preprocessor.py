import re
import string
from typing import List, Tuple

import spacy
from nltk.corpus import stopwords

print("LOADING TEXT PREPROCESSOR")

class TextPreprocessor:
    """
    Reusable NLP preprocessing pipeline.
    """

    def __init__(self):

        # Load spaCy English model
        self.nlp = spacy.load(
            "en_core_web_sm",
            disable=["parser", "ner"]
        )

        # Load stopwords
        self.stop_words = set(stopwords.words("english"))

        # Punctuation table
        self.punctuation_table = str.maketrans(
            "",
            "",
            string.punctuation
        )

    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning.
        """

        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+", "", text)

        # Remove numbers
        text = re.sub(r"\d+", "", text)

        # Remove punctuation
        text = text.translate(self.punctuation_table)

        # Remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def preprocess(
        self,
        text: str
    ) -> Tuple[List[str], str]:
        """
        Full preprocessing pipeline.

        Returns:
            tokens (List[str])
            cleaned_sentence (str)
        """

        # -------------------------
        # Initial cleaning
        # -------------------------
        cleaned_text = self.clean_text(text)

        # -------------------------
        # spaCy processing
        # -------------------------
        doc = self.nlp(cleaned_text)

        tokens = []

        for token in doc:

            lemma = token.lemma_.strip()

            # Skip invalid tokens
            if (
                not lemma
                or lemma in self.stop_words
                or lemma.isspace()
                or len(lemma) <= 1
            ):
                continue

            tokens.append(lemma)

        # Reconstruct cleaned sentence
        cleaned_sentence = " ".join(tokens)

        return tokens, cleaned_sentence

print(TextPreprocessor)