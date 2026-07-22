import numpy as np

class BagOfWordsScratch:
    def __init__(self, max_features=None):
        self.max_features = max_features
        self.vocabulary_ = {}
        self.feature_names_ = []

    def fit(self, raw_documents):
        """
        Learn a vocabulary dictionary of all tokens in the raw documents.
        """
        word_counts = {}
        for doc in raw_documents:
            words = doc.split()
            for word in set(words):
                word_counts[word] = word_counts.get(word, 0) + words.count(word)

        # Sort words by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Limit to max_features
        if self.max_features is not None:
            sorted_words = sorted_words[:self.max_features]
            
        # Create vocabulary mapping
        self.feature_names_ = sorted([word for word, count in sorted_words])
        self.vocabulary_ = {word: idx for idx, word in enumerate(self.feature_names_)}
        return self

    def transform(self, raw_documents):
        """
        Transform documents to document-term matrix.
        """
        n_samples = len(raw_documents)
        n_features = len(self.vocabulary_)
        X = np.zeros((n_samples, n_features), dtype=np.float64)

        for doc_idx, doc in enumerate(raw_documents):
            words = doc.split()
            for word in words:
                if word in self.vocabulary_:
                    X[doc_idx, self.vocabulary_[word]] += 1
        return X

    def fit_transform(self, raw_documents):
        return self.fit(raw_documents).transform(raw_documents)


class TfidfVectorizerScratch:
    def __init__(self, max_features=None, norm='l2', smooth_idf=True):
        self.max_features = max_features
        self.norm = norm
        self.smooth_idf = smooth_idf
        self.bow = BagOfWordsScratch(max_features=max_features)
        self.idf_ = None

    def fit(self, raw_documents):
        """
        Learn vocabulary and idf from training set.
        """
        # Fit Bag of Words first
        self.bow.fit(raw_documents)
        X_bow = self.bow.transform(raw_documents)
        
        n_samples = X_bow.shape[0]
        # Document frequency (number of documents containing term t)
        df = np.sum(X_bow > 0, axis=0)
        
        # IDF Calculation: log((1 + n_samples) / (1 + df)) + 1 if smooth_idf
        if self.smooth_idf:
            self.idf_ = np.log((1 + n_samples) / (1 + df)) + 1
        else:
            # Avoid division by zero by adding condition
            df_safe = np.where(df == 0, 1, df)
            self.idf_ = np.log(n_samples / df_safe) + 1
            
        self.vocabulary_ = self.bow.vocabulary_
        self.feature_names_ = self.bow.feature_names_
        return self

    def transform(self, raw_documents):
        """
        Transform documents to document-term matrix using TF-IDF.
        """
        X_bow = self.bow.transform(raw_documents)
        
        # TF-IDF multiplication
        X_tfidf = X_bow * self.idf_
        
        # L2 Normalization
        if self.norm == 'l2':
            norms = np.linalg.norm(X_tfidf, axis=1, keepdims=True)
            # Avoid division by zero
            norms = np.where(norms == 0, 1e-12, norms)
            X_tfidf = X_tfidf / norms
            
        return X_tfidf

    def fit_transform(self, raw_documents):
        return self.fit(raw_documents).transform(raw_documents)
