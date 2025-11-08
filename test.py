import numpy as np
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase

model = SentenceTransformer("all-MiniLM-L6-v2")
query_emb = model.encode("Which drug was approved for the treatment of chronic cough?")

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "neo4jpass"))
with driver.session() as session:
    docs = session.run("MATCH (d:Document) RETURN d.id AS id, d.text AS text, d.embedding AS emb")
    scored_docs = []
    for row in docs:
        emb = np.array(row["emb"])
        score = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
        scored_docs.append((row["id"], score, row["text"]))
    top = sorted(scored_docs, key=lambda x: x[1], reverse=True)[:3]
    for doc_id, score, text in top:
        print(doc_id, score, text[:80] + "…")

