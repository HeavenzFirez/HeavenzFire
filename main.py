from ai_symphony import AISymphony
from data_api import DataAPI
from knowledge_graph import KnowledgeGraphAPI
import json

class Harmonia:
    def __init__(self, config):
        self.ai_symphony = AISymphony(config["model_path"])
        self.data_api = DataAPI(config["db_path"])
        self.knowledge_graph = KnowledgeGraphAPI(config["kg_path"])

    def process(self, input_data):
        context = self.knowledge_graph.query_nodes("entity")
        model = self.data_api.get_ai_model("model1")
        output = self.ai_symphony.process(input_data)
        return output

if __name__ == "__main__":
    with open('config.json') as config_file:
        config = json.load(config_file)
    harmonia = Harmonia(config)
    input_data = ...  # Define your input data here
    result = harmonia.process(input_data)
    print(result)
