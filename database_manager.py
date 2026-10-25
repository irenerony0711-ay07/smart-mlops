# database_manager.py
import mysql.connector
from pymongo import MongoClient

# Database Configuration (Update password if needed)
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_USER = "viva_user"
MYSQL_PASS = "viva123"  # Put your MySQL password here if you have one
MYSQL_DB = "smart_mlops"

MONGO_URI = "mongodb://localhost:27017"
MONGO_DB = "smart_mlops"

class MongoRegistry:
    def __init__(self):
        self.client = MongoClient(MONGO_URI)
        self.db = self.client[MONGO_DB]
        self.collection = self.db["experiments"]

    def get_leaderboard(self, limit=10):
        # MongoDB Aggregation Pipeline ($sort, $limit, $project)
        pipeline = [
            {"$sort": {"metrics.accuracy": -1}},
            {"$limit": limit},
            {"$project": {"_id": 0, "run_id": 1, "model": 1, "developer": 1, "accuracy": "$metrics.accuracy", "f1_score": "$metrics.f1"}}
        ]
        return list(self.collection.aggregate(pipeline))

    def get_developer_stats(self):
        # MongoDB Aggregation Pipeline ($group, $sum, $max, $avg)
        pipeline = [
            {"$group": {"_id": "$developer", "total_runs": {"$sum": 1}, "max_accuracy": {"$max": "$metrics.accuracy"}, "avg_f1": {"$avg": "$metrics.f1"}}},
            {"$sort": {"total_runs": -1}}
        ]
        return list(self.collection.aggregate(pipeline))


class MySQLVault:
    def __init__(self):
        self.conn = mysql.connector.connect(
            host=MYSQL_HOST, port=MYSQL_PORT,
            user=MYSQL_USER, password=MYSQL_PASS,
            database=MYSQL_DB
        )

    def execute_promotion(self, model_name, version, accuracy, f1_score, developer):
        cursor = self.conn.cursor()
        try:
            cursor.callproc("PromoteModel", [model_name, version, accuracy, f1_score, developer])
            self.conn.commit()
            return True, "Model successfully promoted to Production."
        except mysql.connector.Error as err:
            return False, err.msg
        finally:
            cursor.close()

    def get_active_production_model(self):
        # Nested Subquery Requirement
        cursor = self.conn.cursor(dictionary=True)
        query = """
            SELECT m.model_name, v.version_tag, v.accuracy, v.f1_score, v.registered_at
            FROM model_versions v
            JOIN models m ON v.model_id = m.model_id
            WHERE v.version_id = (SELECT version_id FROM model_versions WHERE is_active_prod = TRUE LIMIT 1);
        """
        cursor.execute(query)
        res = cursor.fetchone()
        cursor.close()
        return res

    def get_release_audit_trail(self):
        # Multi-Table INNER JOIN Requirement
        cursor = self.conn.cursor(dictionary=True)
        query = """
            SELECT h.release_id, m.model_name, v.version_tag, h.promoted_by, h.promotion_status, h.reason, h.promoted_at
            FROM release_history h
            INNER JOIN model_versions v ON h.version_id = v.version_id
            INNER JOIN models m ON v.model_id = m.model_id
            ORDER BY h.promoted_at DESC LIMIT 5;
        """
        cursor.execute(query)
        res = cursor.fetchall()
        cursor.close()
        return res
