import os
import sys

from pymongo import MongoClient

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

import db_config  # noqa: E402

_client = None


def get_mongo_client():
    global _client
    if _client is None:
        _client = MongoClient(db_config.MONGO_URI, serverSelectionTimeoutMS=5000)
    return _client


def get_mongo_collection():
    db = get_mongo_client()[db_config.MONGO_DB]
    return db["experiments"]


def get_leaderboard(limit=10):
    collection = get_mongo_collection()
    pipeline = [
        {"$sort": {"metrics.f1": -1, "metrics.accuracy": -1}},
        {"$limit": int(limit)},
        {
            "$project": {
                "_id": 0,
                "run_id": 1,
                "model": 1,
                "developer": 1,
                "accuracy": "$metrics.accuracy",
                "precision": "$metrics.precision",
                "recall": "$metrics.recall",
                "f1": "$metrics.f1",
                "hyperparameters": 1,
                "timestamp": 1,
            }
        },
    ]
    return list(collection.aggregate(pipeline))


def get_developer_productivity():
    collection = get_mongo_collection()
    pipeline = [
        {
            "$group": {
                "_id": "$developer",
                "total_runs": {"$sum": 1},
                "max_accuracy": {"$max": "$metrics.accuracy"},
                "avg_f1": {"$avg": "$metrics.f1"},
            }
        },
        {"$sort": {"total_runs": -1}},
    ]
    return list(collection.aggregate(pipeline))


def get_average_metrics_by_model():
    collection = get_mongo_collection()
    pipeline = [
        {
            "$group": {
                "_id": "$model",
                "count": {"$sum": 1},
                "avg_accuracy": {"$avg": "$metrics.accuracy"},
                "avg_precision": {"$avg": "$metrics.precision"},
                "avg_recall": {"$avg": "$metrics.recall"},
                "avg_f1": {"$avg": "$metrics.f1"},
            }
        },
        {"$sort": {"avg_f1": -1}},
    ]
    return list(collection.aggregate(pipeline))
