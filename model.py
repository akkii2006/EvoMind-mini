import re
from typing import Dict, List, Tuple
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


class PatternDetector:
    def __init__(self):
        self.patterns = {
            'phone': r'\b(?:\+\d{1,3}[-.\s]?)?\d{10}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        }

    def detect_patterns(self, text: str) -> Dict[str, List[str]]:
        return {
            'phone_numbers': re.findall(self.patterns['phone'], text.lower()),
            'emails': re.findall(self.patterns['email'], text.lower())
        }


class ContentModerationModel:
    def __init__(self):
        self.pattern_detector = PatternDetector()
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.classifier = SVC(kernel='rbf', probability=True)
        self.threshold = 0.75

    def fit(self, texts: List[str], labels: List[int]):
        # Transform texts to TF-IDF features
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)

    def predict(self, texts: List[str]) -> Tuple[np.ndarray, Dict]:
        # Transform texts to TF-IDF features
        X = self.vectorizer.transform(texts)

        # Get probability predictions
        probabilities = self.classifier.predict_proba(X)[:, 1]

        # Detect patterns
        patterns = {}
        for text in texts:
            patterns.update(self.pattern_detector.detect_patterns(text))

        return probabilities, patterns

    def save(self, path: str):
        #Save the model and vectorizer to disk
        model_path = f"{path}/model.joblib"
        vectorizer_path = f"{path}/vectorizer.joblib"

        joblib.dump(self.classifier, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)

    @classmethod
    def load(cls, path: str):
        #Load the model and vectorizer from disk
        model = cls()
        model.classifier = joblib.load(f"{path}/model.joblib")
        model.vectorizer = joblib.load(f"{path}/vectorizer.joblib")
        return model