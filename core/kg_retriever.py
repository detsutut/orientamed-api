from collections import defaultdict
from typing import Union, List

from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
import math
import logging
import os

from core.data_models import RetrievedDocument

logger = logging.getLogger('app.'+__name__)

class KGRetriever:
    def __init__(self, graph_url: Union[str,None]=None, username: Union[str,None]=None, password: Union[str,None]=None):
        if graph_url is not None and username is not None and password is not None:
            self.graph = Neo4jGraph(graph_url, username, password)
        else:
            try:
                self.graph = Neo4jGraph(url=os.getenv("NEO4J_URL"),
                                        username=os.getenv("NEO4J_USR"),
                                        password=os.getenv("NEO4J_PWD"))
            except Exception as e:
                logger.error(e)
                self.graph = None

    def login(self, username: str, password: str, url: Union[str, None]=None):
        url = self.graph_url if url is None else url
        self.graph = Neo4jGraph(url, username, password)

    def get_chunk(self, id: str):
        return self.graph.query("MATCH (n:Chunk) WHERE n.chunkId = $id RETURN n", params={'id': id})[0]['n']

    def _insert_query_node_(self, text, codes):
        cypher = """
            MERGE(q:Query {queryId: 'query0'})
            ON CREATE SET  
                q.text = $text,
                q.concepts = $concepts
            RETURN q    
        """
        self.graph.query(cypher, params={'text': text, 'concepts': codes})

        cypher = """
            MATCH (q:Query), (o:ObjectConcept)
                WHERE o.sctid = $concept
            MERGE (q)-[:HAS_CONCEPT]->(o)
        """
        for c in codes:
            self.graph.query(cypher, params={'concept': str(c)})

    def _shortest_path_bewteen_(self,id1: str, id2: str, max_hops: int = 10):
        cypher = f"""
        MATCH path = shortestPath((initial: ObjectConcept {{id: $id1}})-[*1..{max_hops}]-(final:ObjectConcept {{id: $id2}}))
        WHERE all(n IN nodes(path) WHERE n: ObjectConcept)
        RETURN path
        """

        paths = self.graph.query(cypher, params={'id1': id1, 'id2': id2})
        shortest_path  = list()
        if paths:
            path = paths[0]["path"]  # prendo il percorso (lista di dizionari (nodi) e stringhe (relazioni))
            node_count = math.ceil(len(path) / 2) - 2  # Distanza 0 = nodi direttamente collegati
            shortest_path.append({"id1": id1, "id2": id2, "path": path, "nodeCount": node_count})
        return shortest_path

    def _shortest_path_id_(self,id: str, max_hops: int = 10):
        """Returns a list of shortest paths, one for each chunk"""
        chunk_ids = [c['id'] for c in self.graph.query("MATCH (n:Chunk) RETURN n.chunkId AS id")]

        cypher = f"""
            MATCH path = shortestPath( (start:ObjectConcept {{id: $id}})-[*1..{max_hops}]-(final:Chunk {{chunkId: $chunk_id}}))
            WITH path
            WHERE all(r IN relationships(path) WHERE type(r) <> 'NEXT')
              AND all(n IN nodes(path)[0..-1] WHERE n:ObjectConcept)
            RETURN path
        """

        listPathChunks = list()
        for chunk_id in chunk_ids:
            paths = self.graph.query(cypher, params={'id': id, 'chunk_id': chunk_id})  # result è una lista di dizionari (contiene solo un dizionario con id path)
            if paths:
                path = paths[0]["path"]  # prendo il percorso (lista di dizionari (nodi) e stringhe (relazioni))
                node_count = math.ceil(len(path)/2)-2  # Distanza 0 = nodi direttamente collegati
                listPathChunks.append({"id": chunk_id, "path": path, "nodeCount": node_count})
        return sorted(listPathChunks, key=lambda x: x["nodeCount"])

    def retrieve_average_shortest(self, ids: List[Union[str,int]], max_hops: int = 3):
        logger.info(f"Retrieving Nodes...")
        score_sum = defaultdict(int)
        score_count = defaultdict(int)
        for id in ids:
            connected_chunks = self._shortest_path_id_(id=id, max_hops=max_hops)
            for connected_chunk in connected_chunks:
                score_sum[connected_chunk["id"]] = score_sum.get(connected_chunk["id"],0)+connected_chunk["nodeCount"]
                score_count[connected_chunk["id"]] = score_count.get(connected_chunk["id"],0)+1
        results = [(id,score_sum[id] / score_count[id]) for id in score_sum.keys()]
        sorted_results = sorted(results, key=lambda x: x[1], reverse=False)
        sorted_chunks = []
        for id, score in sorted_results:
            chunk = self.get_chunk(id=id)
            sorted_chunks.append(RetrievedDocument(id=chunk["chunkId"],
                                                    page_content=chunk["text"],
                                                    score=score,
                                                    metadata={"title": chunk["title"],
                                                              "doc_id": chunk["chunkId"],
                                                              "path": None,
                                                              "source": chunk["chunkId"].split("txt")[0]}))
        return sorted_chunks

    def retrieve_absolute_shortest(self, ids: List[Union[str,int]], max_hops: int = 3):
        logger.info(f"Retrieving Nodes...")
        min_scores = defaultdict(int)
        min_paths = defaultdict(str)
        for id in ids:
            connected_chunks = self._shortest_path_id_(id=id, max_hops=max_hops)
            for connected_chunk in connected_chunks:
                id = connected_chunk["id"]
                score = connected_chunk["nodeCount"]
                path = connected_chunk["path"]
                if id not in min_scores:
                    min_scores[id] = score
                else:
                    if score<min_scores[id]:
                        min_scores[id] = score
                        min_paths[id] = path
        sorted_results = sorted(min_scores.items(), key=lambda x: x[1], reverse=False)
        sorted_chunks = []
        for id, score in sorted_results:
            chunk = self.get_chunk(id=id)
            sorted_chunks.append(RetrievedDocument(id=chunk["chunkId"],
                                                    page_content=chunk["text"],
                                                    score=score,
                                                    metadata={"title": chunk["title"],
                                                              "doc_id": chunk["chunkId"],
                                                              "path": min_paths.get(id,None),
                                                              "source": chunk["chunkId"].split("txt")[0]}))
        return sorted_chunks

    def shortest_path_bewteen(self, id1: str, id2: str, max_hops: int = 10):
        cypher = f"""
        MATCH path = shortestPath((initial: ObjectConcept {{id: $id1}})-[*1..{max_hops}]-(final:ObjectConcept {{id: $id2}}))
        WHERE all(n IN nodes(path) WHERE n: ObjectConcept)
        RETURN path
        """

        cypher_backup = f"""
        MATCH (initial:ObjectConcept {{id: $id1}}), (final:ObjectConcept {{id: $id2}})
        MATCH path = (initial)-[*..{max_hops}]-(final)
        WHERE all(r IN relationships(path) WHERE type(r) <> 'NEXT')
        AND all(n IN nodes(path)[0..-1] WHERE n:ObjectConcept)
        RETURN path
        ORDER BY length(path) ASC
        LIMIT 1
        """
        paths = self.graph.query(cypher, params={'id1': id1, 'id2': id2})
        shortest_path  = list()
        if paths:
            path = paths[0]["path"]  # prendo il percorso (lista di dizionari (nodi) e stringhe (relazioni))
            node_count = math.ceil(len(path) / 2) - 2  # Distanza 0 = nodi direttamente collegati
            shortest_path.append({"id1": id1, "id2": id2, "path": path, "nodeCount": node_count})
        return shortest_path


if __name__ == "__main__":
    load_dotenv("secrets.env")
    kg_retriever = KGRetriever()
    retrieved_chunks = kg_retriever.retrieve_average_shortest(["442194005","182353008"],max_hops=5)
    print("AVERAGE SHORTEST")
    print([(chunk.id,chunk.score) for chunk in retrieved_chunks])
    retrieved_chunks = kg_retriever.retrieve_absolute_shortest(["442194005","182353008"],max_hops=5)
    print("ABSOLUTE SHORTEST")
    print([(chunk.id,chunk.score) for chunk in retrieved_chunks])
