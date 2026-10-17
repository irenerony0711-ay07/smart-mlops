from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.tree import DecisionTreeClassifier

from pipeline.preprocess import load_and_preprocess_data
from nosql.mongo_queries import get_mongo_collection


def evaluate_and_log_run(model_instance, model_name, developer, hyperparameters, run_id):
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    model_instance.fit(X_train, y_train)
    predictions = model_instance.predict(X_test)

    accuracy = round(float(accuracy_score(y_test, predictions)), 4)
    precision = round(float(precision_score(y_test, predictions, zero_division=0)), 4)
    recall = round(float(recall_score(y_test, predictions, zero_division=0)), 4)
    f1 = round(float(f1_score(y_test, predictions, zero_division=0)), 4)

    cm = confusion_matrix(y_test, predictions).tolist()

    run_doc = {
        "run_id": run_id,
        "model": model_name,
        "developer": developer,
        "hyperparameters": hyperparameters,
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "confusion_matrix": cm,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    collection = get_mongo_collection()
    collection.replace_one({"run_id": run_id}, run_doc, upsert=True)

    print(
        f"[LOGGED] {run_id} | {model_name} | developer={developer} | "
        f"accuracy={accuracy} | f1={f1}"
    )
    return run_doc


def run_all_training_pipelines():
    evaluate_and_log_run(
        LogisticRegression(C=0.5, max_iter=1000),
        "Logistic Regression",
        "Irene",
        {"C": 0.5, "max_iter": 1000, "solver": "lbfgs"},
        "RUN_010",
    )

    evaluate_and_log_run(
        DecisionTreeClassifier(max_depth=5, min_samples_split=4),
        "Decision Tree",
        "Teammate_1",
        {"max_depth": 5, "min_samples_split": 4, "criterion": "gini"},
        "RUN_011",
    )

    evaluate_and_log_run(
        RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
        "Random Forest",
        "Irene",
        {"n_estimators": 200, "max_depth": 8, "criterion": "gini"},
        "RUN_021",
    )


if __name__ == "__main__":
    run_all_training_pipelines()
