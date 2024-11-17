import sqlite3

class DataAPI:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def get_ai_model(self, model_name):
        self.cursor.execute(f"SELECT model_data FROM ai_models WHERE model_name='{model_name}'")
        return self.cursor.fetchone()[0]
