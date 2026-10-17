import random
from datetime import datetime, timedelta

from nosql.mongo_queries import get_mongo_collection

MODELS = ["Logistic Regression", "Decision Tree", "Random Forest"]
DEVELOPERS = ["Irene", "Teammate_1", "Teammate_2", "Teammate_3"]


def _random_hyperparameters(model_name):
    if model_name == "Logistic Regression":
        return {
            "C": round(random.uniform(0.1, 2.0), 2),
            "max_iter": random.choice([500, 1000, 1500]),
            "solver": random.choice(["lbfgs", "liblinear"]),
        }
    if model_name == "Decision Tree":
        return {
            "max_depth": random.choice([3, 4, 5, 6, 8]),
            "min_samples_split": random.choice([2, 4, 6, 8]),
            "criterion": random.choice(["gini", "entropy"]),
        }
    return {
        "n_estimators": random.choice([100, 150, 200, 300]),
        "max_depth": random.choice([4, 6, 8, 10]),
        "criterion": random.choice(["gini", "entropy"]),
    }


def _build_doc(run_id, timestamp):
    model_name = random.choice(MODELS)
    developer = random.choice(DEVELOPERS)

    accuracy = round(random.uniform(0.68, 0.88), 4)
    precision = round(max(0.0, accuracy - random.uniform(0.0, 0.05)), 4)
    recall = round(max(0.0, accuracy - random.uniform(0.0, 0.05)), 4)
    if precision + recall > 0:
        f1 = round(2 * precision * recall / (precision + recall), 4)
    else:
        f1 = 0.0

    return {
        "run_id": run_id,
        "model": model_name,
        "developer": developer,
        "hyperparameters": _random_hyperparameters(model_name),
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "confusion_matrix": [[50, 20], [15, 115]],
        "timestamp": timestamp.isoformat() + "Z",
    }


def seed_past_runs():
    random.seed(42)
    collection = get_mongo_collection()

    now = datetime.utcnow()
    for index in range(1, 16):
        run_id = f"RUN_{index:03d}"
        days_ago = random.randint(1, 20)
        timestamp = now - timedelta(
            days=days_ago,
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
        )
        doc = _build_doc(run_id, timestamp)
        collection.replace_one({"run_id": run_id}, doc, upsert=True)

    print("Pre-seeded 15 benchmark experiment runs in MongoDB.")


if __name__ == "__main__":
    seed_past_runs()
