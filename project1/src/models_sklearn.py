from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def get_sklearn_models():
    """
    Returns baseline scikit-learn models matching the provided skeleton.
    """
    return {
        "KNN": KNeighborsClassifier(n_neighbors=5, metric='cosine'),
        "LogReg": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "NeuralNet": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    }
