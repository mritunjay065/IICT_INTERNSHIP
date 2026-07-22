import numpy as np

# =====================================================================
# 1. K-Nearest Neighbors Classifier from Scratch
# =====================================================================
class KNNClassifierScratch:
    def __init__(self, n_neighbors=5, metric='cosine'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self

    def predict(self, X):
        X = np.array(X)
        preds = []
        for x in X:
            if self.metric == 'cosine':
                # Cosine similarity: (a . b) / (||a|| * ||b||)
                # Cosine distance = 1 - Cosine similarity
                # Avoid division by zero
                norms = np.linalg.norm(self.X_train, axis=1) * np.linalg.norm(x)
                norms = np.where(norms == 0, 1e-12, norms)
                similarities = np.dot(self.X_train, x) / norms
                distances = 1.0 - similarities
            else:
                # Euclidean distance
                distances = np.linalg.norm(self.X_train - x, axis=1)
            
            # Find the indices of the k smallest distances
            k_indices = np.argsort(distances)[:self.n_neighbors]
            k_nearest_labels = self.y_train[k_indices]
            
            # Majority vote
            counts = np.bincount(k_nearest_labels)
            preds.append(np.argmax(counts))
            
        return np.array(preds)


# =====================================================================
# 2. Logistic Regression Classifier from Scratch
# =====================================================================
class LogisticRegressionScratch:
    def __init__(self, lr=0.1, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        # Clip z to prevent overflow/underflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        
        # Init weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        # Gradient descent
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            
            # Gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return self

    def predict_proba(self, X):
        X = np.array(X)
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.array([1 if p >= 0.5 else 0 for p in proba])


# =====================================================================
# 3. Decision Tree and Random Forest Classifier from Scratch
# =====================================================================
class DecisionNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature          # Index of feature to split on
        self.threshold = threshold      # Threshold value for split
        self.left = left                # Left subtree
        self.right = right              # Right subtree
        self.value = value              # Classification value if leaf node

    def is_leaf_node(self):
        return self.value is not None


class DecisionTreeScratch:
    def __init__(self, min_samples_split=2, max_depth=10, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)
        return self

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Check stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return DecisionNode(value=leaf_value)

        # Find best split
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)

        # Create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            leaf_value = self._most_common_label(y)
            return DecisionNode(value=leaf_value)

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return DecisionNode(feature=best_feat, threshold=best_thresh, left=left, right=right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = 0, 0.0

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            # To speed up tree construction for large vocabulary size, sample up to 10 thresholds
            if len(thresholds) > 10:
                thresholds = np.random.choice(thresholds, 10, replace=False)
                
            for thresh in thresholds:
                gain = self._information_gain(y, X_column, thresh)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thresh

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, thresh):
        # Parent Gini
        parent_gini = self._gini(y)
        
        # Split indexes
        left_idxs, right_idxs = self._split(X_column, thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # Weighted average child Gini
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        gini_l, gini_r = self._gini(y[left_idxs]), self._gini(y[right_idxs])
        child_gini = (n_l / n) * gini_l + (n_r / n) * gini_r
        
        # Info Gain = parent Gini - weighted child Gini
        return parent_gini - child_gini

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _gini(self, y):
        m = len(y)
        if m == 0:
            return 0
        counts = np.bincount(y)
        probabilities = counts / m
        return 1.0 - np.sum(probabilities ** 2)

    def _most_common_label(self, y):
        if len(y) == 0:
            return 0
        return np.argmax(np.bincount(y))

    def predict(self, X):
        X = np.array(X)
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class RandomForestScratch:
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.trees = []
        n_samples = X.shape[0]
        
        # Calculate feature subsample size if max_features is specified
        if self.max_features is None:
            n_features_sub = int(np.sqrt(X.shape[1])) # standard for classification
        else:
            n_features_sub = self.max_features

        for _ in range(self.n_estimators):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]
            
            # Create and fit tree
            tree = DecisionTreeScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=n_features_sub
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        return self

    def predict(self, X):
        X = np.array(X)
        # Collect predictions from all trees
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Majority vote across estimators
        # tree_preds has shape (n_estimators, n_samples)
        # Transpose to (n_samples, n_estimators)
        tree_preds = tree_preds.T
        preds = []
        for sample_preds in tree_preds:
            counts = np.bincount(sample_preds)
            preds.append(np.argmax(counts))
        return np.array(preds)


# =====================================================================
# 4. Simple Neural Network (Multilayer Perceptron) from Scratch
# =====================================================================
class NeuralNetworkScratch:
    def __init__(self, input_size, hidden_size=64, lr=0.1, epochs=100, batch_size=32):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Weight initialization (He Initialization for hidden, Xavier for output)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, 1) * np.sqrt(1.0 / hidden_size)
        self.b2 = np.zeros((1, 1))
        
        # For tracking training loss
        self.loss_history = []

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_derivative(self, a):
        return (a > 0).astype(float)

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _binary_cross_entropy(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        n_samples = X.shape[0]
        
        for epoch in range(self.epochs):
            # Shuffle indices
            shuffled_indices = np.random.permutation(n_samples)
            X_shuffled = X[shuffled_indices]
            y_shuffled = y[shuffled_indices]
            
            epoch_loss = 0.0
            num_batches = int(np.ceil(n_samples / self.batch_size))
            
            for b in range(num_batches):
                start_idx = b * self.batch_size
                end_idx = min(start_idx + self.batch_size, n_samples)
                
                xb = X_shuffled[start_idx:end_idx]
                yb = y_shuffled[start_idx:end_idx]
                
                # --- Forward Pass ---
                z1 = np.dot(xb, self.W1) + self.b1
                a1 = self._relu(z1)
                
                z2 = np.dot(a1, self.W2) + self.b2
                a2 = self._sigmoid(z2)
                
                # --- Backward Pass ---
                m = xb.shape[0]
                
                # Output layer gradient
                dz2 = a2 - yb
                dW2 = (1.0 / m) * np.dot(a1.T, dz2)
                db2 = (1.0 / m) * np.sum(dz2, axis=0, keepdims=True)
                
                # Hidden layer gradient
                da1 = np.dot(dz2, self.W2.T)
                dz1 = da1 * self._relu_derivative(a1)
                dW1 = (1.0 / m) * np.dot(xb.T, dz1)
                db1 = (1.0 / m) * np.sum(dz1, axis=0, keepdims=True)
                
                # --- Weight Updates ---
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
                
            # Track epoch loss
            z1_full = np.dot(X, self.W1) + self.b1
            a1_full = self._relu(z1_full)
            z2_full = np.dot(a1_full, self.W2) + self.b2
            a2_full = self._sigmoid(z2_full)
            loss = self._binary_cross_entropy(y, a2_full)
            self.loss_history.append(loss)
            
        return self

    def predict_proba(self, X):
        X = np.array(X)
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self._relu(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        return self._sigmoid(z2)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int).flatten()
