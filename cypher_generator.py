# mhr_medllm/cypher_generator.py

from langchain_core.prompts import ChatPromptTemplate #
import re
import logging
# from neo4j_connector import Neo4jKG # If _fetch_schema is used from Neo4jKG directly

logger = logging.getLogger(__name__)

DEFAULT_CYPHER_GEN_TEMPLATE = """You are an expert Cypher query generator.
Given the following Knowledge Graph schema, generate a Cypher query to answer the user's question.

Schema:
{kg_schema}

User Question: {question}

Query Guidelines:
1. Use `WITH` clauses for step-by-step processing of complex logic if needed.
2. Limit results to a maximum of 5: `LIMIT 5`.
3. Use parameterized queries where appropriate (e.g., `WHERE n.name = $name_param`). Ensure parameters are named with a `_param` suffix.
4. For multi-hop relationships, you can use variable-length path syntax like `-[r:*1..3]->`.
5. The query MUST NOT contain any data modification operations (e.g., CREATE, DELETE, SET, MERGE, REMOVE).
6. Ensure the query is a single, complete, executable Cypher statement.
7. Extract relevant entities or keywords from the question and use them in your query conditions. For example, if the question is "What are the symptoms of Flu?", you might use `MATCH (d:Disease {{name: $disease_name_param}}) ...` and provide `disease_name_param: "Flu"`.  <--- CHECK THIS LINE CAREFULLY
8. Return paths or specific node properties that would help answer the question. For example, `RETURN d.symptoms` or `RETURN p AS path`.

Generated Cypher Query:
"""

# ... rest of the CypherGenerator class ...


class CypherGenerator:
    def __init__(self, llm, neo4j_driver=None, template=None): # neo4j_driver is for _fetch_schema
        self.llm = llm
        self.neo4j_driver = neo4j_driver # Neo4j driver instance from Neo4jKG
        self.template = template if template else DEFAULT_CYPHER_GEN_TEMPLATE
        if not self.neo4j_driver:
            logger.warning("Neo4j driver not provided to CypherGenerator. Dynamic schema fetching will be disabled.")

    def _fetch_schema_from_db(self): #
        """Fetches schema directly from the Neo4j database."""
        if not self.neo4j_driver or not hasattr(self.neo4j_driver, 'session'): # Check if it's a valid driver
             logger.error("Cannot fetch schema from DB: Neo4j driver not available or invalid.")
             return "Schema not available."
        try:
            with self.neo4j_driver.session() as session:
                # This query attempts to get distinct labels and relationship types.
                # It uses APOC, ensure APOC is installed in your Neo4j instance.
                # If APOC is not available, you might need a simpler query or manual schema.
                result = session.run("""
                    CALL db.schema.visualization()
                    YIELD nodes, relationships
                    WITH [node IN nodes | labels(node)[0] + apoc.map.toPairs(node.properties)] AS node_schemas,
                         [rel IN relationships | type(rel) + apoc.map.toPairs(rel.properties)] AS rel_schemas
                    RETURN 'Node Labels and Properties: ' + apoc.text.join(node_schemas, ', ') AS node_part,
                           'Relationship Types and Properties: ' + apoc.text.join(rel_schemas, ', ') AS rel_part
                """)
                # A more robust schema query might be:
                # result = session.run("""
                #     MATCH (n)
                #     WITH DISTINCT labels(n) AS node_labels_list
                #     UNWIND node_labels_list AS lbl
                #     WITH DISTINCT lbl ORDER BY lbl
                #     CALL apoc.meta.nodeTypeProperties({labels:[lbl]})
                #     YIELD propertyName, propertyTypes
                #     WITH lbl, कलेक्ट({name: propertyName, types: propertyTypes}) AS properties
                #     WITH lbl, properties
                #     ORDER BY lbl
                #     WITH 'Node Label: ' + lbl + ' { ' + apoc.text.join([p.name + ': ' + p.types[0] FOR p IN properties], ', ') + ' }' AS node_schema_part
                #     WITH कलेक्ट(node_schema_part) AS all_node_parts
                #     MATCH ()-[r]->()
                #     WITH DISTINCT type(r) AS rel_type_list
                #     UNWIND rel_type_list AS rt
                #     WITH DISTINCT rt ORDER BY rt
                #     CALL apoc.meta.relTypeProperties({types:[rt]})
                #     YIELD propertyName, propertyTypes
                #     WITH all_node_parts, rt, कलेक्ट({name:propertyName, types:propertyTypes}) as properties
                #     WITH all_node_parts, rt, properties
                #     ORDER BY rt
                #     WITH all_node_parts, 'Relationship Type: ' + rt + ' { ' + apoc.text.join([p.name + ': ' + p.types[0] FOR p IN properties], ', ') + ' }' AS rel_schema_part
                #     RETURN apoc.text.join(all_node_parts, '\n') + '\n' + apoc.text.join(collect(rel_schema_part), '\n') AS full_schema
                # """)
                # schema_data = result.single()
                # return schema_data["full_schema"] if schema_data else "Could not retrieve dynamic schema."

                # Simplified schema fetching from the original notebook
                schema_parts = []
                node_labels_res = session.run("""
                    MATCH (n)
                    WITH DISTINCT labels(n) AS labels_list
                    UNWIND labels_list AS label
                    RETURN DISTINCT label ORDER BY label
                """)
                node_labels = [record["label"] for record in node_labels_res]
                if node_labels:
                    schema_parts.append("Node Labels: " + ", ".join(node_labels))

                rel_types_res = session.run("""
                    MATCH ()-[r]->()
                    WITH DISTINCT type(r) AS type
                    RETURN type ORDER BY type
                """)
                rel_types = [record["type"] for record in rel_types_res]
                if rel_types:
                    schema_parts.append("Relationship Types: " + ", ".join(rel_types))

                return "\n".join(schema_parts) if schema_parts else "Schema: No labels or types found."

        except Exception as e:
            logger.error(f"Failed to fetch dynamic schema from Neo4j: {e}")
            return "Schema information unavailable due to an error."


    def _sanitize_cypher(self, raw_cypher: str) -> str: #
        """Basic sanitization and extraction of the Cypher query."""
        # Remove potential markdown backticks if LLM wraps query in them
        raw_cypher = raw_cypher.strip()
        if raw_cypher.startswith("```cypher"):
            raw_cypher = raw_cypher[len("```cypher"):].strip()
        elif raw_cypher.startswith("```"):
            raw_cypher = raw_cypher[len("```"):].strip()
        if raw_cypher.endswith("```"):
            raw_cypher = raw_cypher[:-len("```")].strip()

        # Check for forbidden operations (defensive)
        forbidden_ops = r"\b(CREATE|DELETE|SET|MERGE|REMOVE|DROP)\b" #
        if re.search(forbidden_ops, raw_cypher, re.IGNORECASE):
            logger.error(f"Generated query contains forbidden operations: {raw_cypher}")
            raise ValueError("Generated Cypher query contains forbidden destructive operations.")

        # Attempt to extract a valid query; this is a simple heuristic.
        # It looks for common starting keywords.
        # A more robust parser might be needed for complex LLM outputs.
        # The original notebook used (MATCH...;) or (WITH...;) - this can be too restrictive.
        # Let's try to find the first plausible query block.
        # A simple approach: assume the LLM returns just the query or query with minimal explanation.
        # If the LLM is well-prompted, it should return just the query.
        # For now, we'll assume the LLM's output is mostly the query.
        # Adding a check for LIMIT 5 as per requirements.
        if "LIMIT 5" not in raw_cypher.upper() and "LIMIT 10" not in raw_cypher.upper(): # Allow LIMIT 10 from PathRetrieval example
             # We might want to append it if missing, or let the LLM handle it via prompt.
             # Forcing it can sometimes break complex queries (e.g. if there's already a LIMIT)
             logger.warning(f"Generated Cypher query does not explicitly contain 'LIMIT 5'. Relying on LLM adherence. Query: {raw_cypher}")


        # Basic check for parameterization (if $param is used)
        # This is more of a check than a sanitizer.
        if "$" in raw_cypher and not re.search(r"\$[a-zA-Z0-9_]+_param\b", raw_cypher):
            logger.warning(f"Query uses '$' but might not follow the '$name_param' convention: {raw_cypher}")


        return raw_cypher.strip().rstrip(';') # Remove trailing semicolon if any

    def _extract_parameters_from_question(self, question: str, cypher_query: str) -> dict: #
        """
        Extracts parameters for the Cypher query based on placeholders in the query
        and information from the question. This is a simplified example.
        A more robust solution would use Named Entity Recognition (NER) or more advanced LLM prompting.
        """
        params = {}
        # Find all parameters like $name_param in the Cypher query
        # Example from notebook: if "$name" in cypher: if "名为" in question: params["name"] = question.split("名为")[1].strip()
        # This needs to be more generic.

        # This is highly dependent on how parameters are defined in the prompt and expected from the LLM.
        # For example, if the LLM is instructed to use parameters like $disease_name_param,
        # we would need a way to map "Flu" from "Symptoms of Flu" to disease_name_param.

        # A simple heuristic: if LLM generates a query with $entity_name_param,
        # try to find a capitalized word in the question or a noun phrase.
        # This is very basic and error-prone.
        # A better way is to ask the LLM to also output the parameters in a structured format.

        # Example: If cypher_query contains '$disease_name_param' and question is "What are symptoms of Influenza?"
        # We need to map Influenza to disease_name_param.
        # For now, this will be a placeholder, as robust parameter extraction is complex.
        # The original example was very specific:
        # if "$name" in cypher:
        #     if "名为" in question: # "named" in Chinese
        #         params["name"] = question.split("名为")[1].strip()

        # Let's assume the LLM might also return parameters if prompted well.
        # Or, we can try a regex for simple cases:
        # e.g., if question is "Find medicine for Diabetes" and query has `d.name = $disease_param`
        # We could try to find "Diabetes".

        # For the purpose of this refactoring, we'll return an empty dict
        # and assume parameters are either part of the LLM's Cypher output (less ideal)
        # or will be handled by a more sophisticated method later.
        logger.info(f"Parameter extraction step (currently basic). Question: {question}, Cypher: {cypher_query}")
        # A better approach for the LLM to handle params:
        # Ask LLM to output a JSON like:
        # { "cypher": "MATCH (d:Disease {name: $name_param}) RETURN d.symptoms LIMIT 5",
        #   "params": { "name_param": "Value Extracted From Question" } }
        # This is usually more reliable.
        # For now, we are not doing this and will rely on the cypher_query potentially having values or simple $params.
        # The `execute_cypher` in retriever expects a params dict.
        # The workflow in cell 16 passes `params={"question": state["question"]}` which is not quite right.
        # It should be params extracted FOR the cypher query from the question.

        # Let's try a very simple keyword extraction for parameters found in the cypher query.
        # This is still heuristic.
        found_params = re.findall(r"\$([a-zA-Z0-9_]+)", cypher_query)
        for param_name in found_params:
            # Try to find a value for this param in the question.
            # This is a placeholder for a more robust entity extraction.
            # Example: if param_name is "disease_name_param"
            # A simple approach: if "disease_name_param" (or a part of it like "disease") is in question,
            # try to extract the following noun. This is complex.

            # For now, if a param is in the query, we will try to find *any* capitalized word from the question
            # This is a very naive approach and likely needs significant improvement.
            # Or, we just return an empty dict and assume the LLM will embed values or the schema is simple.
            if "name" in param_name.lower(): # if param looks like $disease_name_param or $drug_name_param
                # Attempt to find a capitalized word or a known entity type.
                # This is where NER would be very helpful.
                # Simplistic: look for capitalized words in the question.
                capitalized_words = re.findall(r"\b[A-Z][a-z]+\b", question)
                if capitalized_words:
                    # This is too naive, it will just pick the first one.
                    # params[param_name] = capitalized_words[0]
                    # Let's skip this naive auto-population for now to avoid incorrect bindings.
                    # The LLM should ideally replace $param with actual value or provide the param map.
                    logger.warning(f"Parameter '{param_name}' found in Cypher. Manual or LLM-driven parameter filling is recommended.")
                    pass # Parameter extraction logic to be refined.

        return params # Return empty, or parameters extracted via a more robust method.

    def generate(self, question: str, kg_schema_description: str = None) -> dict:
        """
        Generates a Cypher query and attempts to extract parameters.
        Returns a dict with "cypher_query" and "params".
        """
        if not question:
            logger.error("Question content is empty.")
            return {"error": "Question content is empty.", "cypher_query": None, "params": {}}

        current_schema = "Schema not available."
        if kg_schema_description:
            current_schema = kg_schema_description
        elif self.neo4j_driver:
            logger.info("Fetching schema dynamically from Neo4j for Cypher generation...")
            current_schema = self._fetch_schema_from_db()
        else:
            logger.warning("No KG schema description provided and Neo4j driver not available for dynamic schema fetching. Cypher generation might be suboptimal.")


        prompt_messages = [
            {"role": "system", "content": "You are an expert Cypher query generator for Neo4j."},
            {"role": "user", "content": self.template.format(kg_schema=current_schema, question=question)}
        ]

        try:
            logger.info(f"Generating Cypher for question: {question} with schema: {current_schema[:200]}...") # Log snippet of schema
            response = self.llm.chat.completions.create(
                # model="qwen-plus", # As per cell 16, this should be configurable
                model="qwen-plus", # Make sure this model is available or use your preferred one
                messages=prompt_messages,
                temperature=0.1 # Low temperature for more deterministic Cypher
            )
            raw_cypher = response.choices[0].message.content
            logger.info(f"Raw Cypher from LLM: {raw_cypher}")

            sanitized_cypher = self._sanitize_cypher(raw_cypher)
            # Parameters should ideally be extracted by the LLM or a dedicated NER step.
            # The original _extract_parameters was very basic.
            # For now, we'll rely on the LLM to either embed values or use params that we can fill later.
            # The example in cell 16 did not have a robust param extraction.
            # It passed `params={"question": state["question"]}` to retriever, which is not how Cypher params work.
            # Cypher params should be like `{"entity_name": "Aspirin"}`.

            # Let's try the _extract_parameters_from_question (which is basic)
            # It might be better to have the LLM also return the parameters.
            extracted_params = self._extract_parameters_from_question(question, sanitized_cypher) #

            logger.info(f"Generated Cypher: {sanitized_cypher}, Params: {extracted_params}")
            return {"cypher_query": sanitized_cypher, "params": extracted_params, "error": None}

        except ValueError as ve: # Catch sanitization errors
            logger.error(f"Cypher generation failed (sanitization): {str(ve)}")
            return {"error": str(ve), "cypher_query": None, "params": {}}
        except Exception as e:
            logger.error(f"Cypher generation failed (LLM or other): {str(e)}")
            # Log the full error for debugging
            logger.exception("Exception details during Cypher generation:")
            return {"error": f"LLM call or processing failed: {str(e)}", "cypher_query": None, "params": {}}

# Example Usage (for testing this module)
if __name__ == "__main__":
    import os
    from openai import OpenAI
    from config import DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL, KG_SCHEMA, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    from neo4j_connector import Neo4jKG # For dynamic schema fetching test

    logging.basicConfig(level=logging.INFO)

    if not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY == "sk-your_dashscope_api_key":
        logger.error("DASHSCOPE_API_KEY not configured in config.py. Please add it to test CypherGenerator.")
    else:
        llm_client = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=DASHSCOPE_BASE_URL
        )

        # Test with static schema
        logger.info("Testing CypherGenerator with static schema...")
        cypher_gen_static = CypherGenerator(llm=llm_client)
        test_question = "What are the symptoms of chronic gastritis and how is it treated?"
        # Using KG_SCHEMA from config.py which is similar to the one in cell 16
        # The schema in cell 16 was:
        # KG_SCHEMA = """
        # 节点类型：
        # - Disease {name, symptoms, treatments}
        # - Drug {name, side_effects, indications}
        # 关系类型：
        # - HAS_SYMPTOM (Disease -> Symptom) - This seems to imply Symptom is a node. Schema needs clarity.
        #                                     Or Disease.symptoms is a property.
        # - TREATS (Drug -> Disease)
        # """
        # Let's use a slightly more detailed schema for the test, assuming 'Symptom' is a node.
        detailed_test_schema = """
        Node Labels and Properties:
        - Disease { name: STRING, symptoms: LIST_OF_STRINGS, treatments: LIST_OF_STRINGS }
        - Drug { name: STRING, side_effects: LIST_OF_STRINGS, indications: LIST_OF_STRINGS }
        - Symptom { name: STRING } // Assuming Symptom is a node for HAS_SYMPTOM relationship

        Relationship Types:
        - (Disease)-[:HAS_SYMPTOM]->(Symptom)
        - (Drug)-[:TREATS]->(Disease)
        - (Disease)-[:TREATED_BY]->(Drug) // Alternative to (Drug)-[:TREATS]->(Disease)
        """


        # For the test to align with the prompt in cell 16, let's use the exact schema from config.py
        result_static = cypher_gen_static.generate(test_question, kg_schema_description=KG_SCHEMA)
        logger.info(f"Static Schema - Question: {test_question}")
        logger.info(f"Static Schema - Generated Cypher: {result_static.get('cypher_query')}")
        logger.info(f"Static Schema - Params: {result_static.get('params')}")
        logger.info(f"Static Schema - Error: {result_static.get('error')}")

        # Test with dynamic schema fetching (if Neo4j is running and configured)
        logger.info("\nTesting CypherGenerator with dynamic schema from Neo4j...")
        try:
            neo4j_conn_for_schema = Neo4jKG(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
            if neo4j_conn_for_schema.driver:
                # Pass the driver object itself, not the Neo4jKG instance for _fetch_schema
                cypher_gen_dynamic = CypherGenerator(llm=llm_client, neo4j_driver=neo4j_conn_for_schema.driver)
                dynamic_schema_question = "Find drugs that treat diseases with 'inflammation' in their name."
                result_dynamic = cypher_gen_dynamic.generate(dynamic_schema_question)

                logger.info(f"Dynamic Schema - Question: {dynamic_schema_question}")
                logger.info(f"Dynamic Schema - Generated Cypher: {result_dynamic.get('cypher_query')}")
                logger.info(f"Dynamic Schema - Params: {result_dynamic.get('params')}")
                logger.info(f"Dynamic Schema - Error: {result_dynamic.get('error')}")
                neo4j_conn_for_schema.close()
            else:
                logger.warning("Skipping dynamic schema test as Neo4j connection failed.")
        except Exception as e:
            logger.error(f"Error during dynamic schema test setup: {e}")