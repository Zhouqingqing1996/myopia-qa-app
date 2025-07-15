# mhr_medllm/neo4j_connector.py

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Neo4jKG:
    def __init__(self, uri, user, password):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j.")
            # SentenceTransformer model can be loaded here or passed as an argument if used by other components
            # self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.driver = None # Ensure driver is None if connection fails

    def close(self):
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed.")

    def execute_read(self, query, params=None):
        """Executes a read-only Cypher query."""
        if not self.driver:
            logger.error("Neo4j driver not initialized.")
            return []
        try:
            with self.driver.session() as session:
                result = session.execute_read(lambda tx: list(tx.run(query, params)))
                return result
        except Exception as e:
            logger.error(f"Neo4j read query failed: {query} | PARAMS: {params} | ERROR: {e}")
            return []

    def execute_write(self, query, params=None):
        """Executes a write Cypher query."""
        if not self.driver:
            logger.error("Neo4j driver not initialized.")
            return []
        try:
            with self.driver.session() as session:
                result = session.execute_write(lambda tx: list(tx.run(query, params)))
                return result
        except Exception as e:
            logger.error(f"Neo4j write query failed: {query} | PARAMS: {params} | ERROR: {e}")
            return []

    def initialize_indexes(self): # Adapted from cell 4
        """Initializes full-text and potentially vector indexes."""
        if not self.driver:
            logger.error("Neo4j driver not initialized. Cannot create indexes.")
            return

        # Full-text index for Entity nodes
        # This assumes 'Entity' is a common label you want to search on.
        # Adjust 'Entity', 'name', 'description', 'aliases' as per your graph schema.
        entity_search_query = """
        CREATE FULLTEXT INDEX entitySearch IF NOT EXISTS
        FOR (n:Entity)
        ON EACH [n.name, n.description, n.aliases]
        """
        try:
            self.execute_write(entity_search_query)
            logger.info("Full-text index 'entitySearch' created or already exists.")
        except Exception as e:
            logger.error(f"Failed to create full-text index 'entitySearch': {e}")

        # Example for creating a vector index if you store embeddings and use Neo4j's vector capabilities
        # This is a conceptual example; syntax might vary based on Neo4j version and plugins.
        # Requires a property named 'embedding' on nodes with label 'Entity'.
        # vector_index_query = """
        # CREATE VECTOR INDEX `entityVectorIndex` IF NOT EXISTS
        # FOR (n:Entity) ON (n.embedding)
        # OPTIONS {indexConfig: {
        #  `vector.dimensions`: 384,  // Dimension of 'all-MiniLM-L6-v2'
        #  `vector.similarity_function`: 'cosine'
        # }}
        # """
        # try:
        # self.execute_write(vector_index_query)
        #     logger.info("Vector index 'entityVectorIndex' created or already exists.")
        # except Exception as e:
        #     logger.error(f"Failed to create vector index 'entityVectorIndex': {e}")


    # Method to generate and store embeddings if needed (adapted from cell 4)
    # This is a heavy operation and might be better suited for a separate script or offline process.
    def generate_and_store_embeddings(self, model_name='all-MiniLM-L6-v2'):
        if not self.driver:
            logger.error("Neo4j driver not initialized. Cannot generate embeddings.")
            return

        logger.info(f"Loading sentence transformer model: {model_name}")
        model = SentenceTransformer(model_name)

        logger.info("Fetching Entity nodes for embedding generation...")
        # Adjust the query to fetch relevant text properties for embedding
        result = self.execute_read("MATCH (e:Entity) WHERE e.name IS NOT NULL AND e.description IS NOT NULL RETURN id(e) AS node_id, e.name as name, e.description as description")

        if not result:
            logger.info("No Entity nodes found to embed or missing 'name'/'description' properties.")
            return

        texts_to_embed = []
        node_ids = []
        for record in result:
            texts_to_embed.append(f"{record['name']} {record['description']}")
            node_ids.append(record['node_id'])

        if texts_to_embed:
            logger.info(f"Generating embeddings for {len(texts_to_embed)} entities...")
            embeddings = model.encode(texts_to_embed, convert_to_numpy=True, show_progress_bar=True)
            logger.info("Embeddings generated. Storing them back to Neo4j...")

            for i, embedding in enumerate(embeddings):
                node_id = node_ids[i]
                self.execute_write(
                    "MATCH (e) WHERE id(e) = $node_id SET e.embedding = $embedding",
                    {"node_id": node_id, "embedding": embedding.tolist()}
                )
            logger.info("Embeddings stored successfully.")
        else:
            logger.info("No texts to embed.")

# Example usage (typically not in this file, but for testing)
if __name__ == "__main__":
    from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    logging.basicConfig(level=logging.INFO)

    # Test connection
    neo4j_conn = Neo4jKG(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    if neo4j_conn.driver:
        print("Testing connection by fetching node count:")
        count_result = neo4j_conn.execute_read("MATCH (n) RETURN count(n) AS count")
        if count_result:
            print(f"Node count: {count_result[0]['count']}")
        else:
            print("Failed to fetch node count.")

        # Initialize indexes (run once or when schema changes)
        # print("Initializing indexes...")
        # neo4j_conn.initialize_indexes()

        # Generate and store embeddings (run once or when data changes significantly)
        # print("Generating and storing embeddings (this might take a while)...")
        # neo4j_conn.generate_and_store_embeddings() # Ensure 'Entity' nodes with 'name' and 'description' exist

        neo4j_conn.close()
    else:
        print("Could not establish Neo4j connection for testing.")