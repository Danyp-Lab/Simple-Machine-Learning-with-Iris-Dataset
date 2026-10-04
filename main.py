"""
Simple Machine Learning with Iris Dataset
Modernized for PEP standards with type annotations and clean entry point.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def train_and_evaluate(test_size: float = 0.3, random_state: int = 42, n_neighbors: int = 3) -> float:
    """Load Iris dataset, fit a KNN classifier, and return accuracy score."""
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=test_size, random_state=random_state
    )

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(x_train, y_train)

    accuracy: float = float(knn.score(x_test, y_test))
    return accuracy


def main() -> None:
    accuracy = train_and_evaluate()
    print(f"✅ Model trained successfully. Test Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
