from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_and_save_model():
    # Iris veri setini yükleme
    iris_data = load_iris()
    X_iris = iris_data.data
    y_iris = iris_data.target

    # Rastgele Orman sınıflandırıcı modelini eğitme
    rf_iris = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_iris.fit(X_iris, y_iris)

    # Modeli kaydetme
    iris_model_path = "data/iris_model.pkl"
    joblib.dump(rf_iris, iris_model_path)

    return rf_iris,iris_data
