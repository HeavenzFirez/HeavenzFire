import networkx as nx

class KnowledgeGraphAPI:
    def __init__(self, kg_path):
        self.kg = nx.read_gpickle(kg_path)

    def query_nodes(self, node_type):
        return [node for node in self.kg.nodes(data=True) if node[1]["type"] == node_type]
