"""
Pipeline script for the D2GEP (Document‑to‑Graph‑and‑Embeddings Pipeline).

This script demonstrates how to ingest raw text documents, split them into
manageable chunks, extract entities and relationships using a large language
model, store the resulting knowledge graph in Neo4j, and generate
embeddings for each document chunk.  It is intended to be run inside the
`pipeline` service defined in the accompanying docker‑compose.yml file.

Usage:
  ```bash
  docker compose up
  ```

The `pipeline` service will install dependencies and execute this script
automatically.  Alternatively, you can run it manually with:
  ```bash
  python pipeline/pipeline.py
  ```

Note: This sample is for demonstration purposes.  It uses a small
SentenceTransformer model for embeddings and assumes a trivial mapping
between document chunks and graph nodes.  In production you should tune
the chunk size, schema, prompts and embedding strategy to fit your data.
"""

import os
from typing import List, Tuple

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import spacy
from spacy.pipeline import EntityRuler
from sentence_transformers import SentenceTransformer
from langchain_community.graphs import Neo4jGraph


def load_documents(path: str):
    """Load a plain‑text document from the given path.

    Returns a list of Document objects compatible with LangChain.
    """
    loader = TextLoader(path)
    return loader.load()


def chunk_documents(documents: List, chunk_size: int = 1024, overlap: int = 128):
    """Split documents into overlapping chunks using LangChain.

    Parameters
    ----------
    documents : list of Document
        Documents loaded by a LangChain loader.
    chunk_size : int
        Maximum number of characters per chunk.
    overlap : int
        Number of characters of overlap between consecutive chunks.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(documents)


def extract_entities_spacy(text: str, nlp) -> List[Tuple[str, str]]:
    """Perform NER using spaCy and return a list of (entity, label) tuples."""
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


def create_graph_from_chunks(chunks, nlp, graph):
    """Create Document and Entity nodes in Neo4j from text chunks.

    For each chunk, a Document node is created with properties `id` and `text`.
    Named entities extracted via spaCy are merged into Entity nodes with a
    `name` property (unique) and a `type` property for the NER label.  A
    MENTIONS relationship connects Document nodes to the entities they contain.
    """
    for idx, chunk in enumerate(chunks):
        text = chunk.page_content
        ents = extract_entities_spacy(text, nlp)
        # Build list of dictionaries for Cypher
        ent_list = []
        for name, label in ents:
            ent_list.append({"name": name, "type": label})
        # Run a Cypher query to create/merge nodes and relationships
        cypher = """
        MERGE (d:Document {id: $idx})
        SET d.text = $text
        WITH d
        UNWIND $ents AS ent
        MERGE (e:Entity {name: ent.name})
        ON CREATE SET e.type = ent.type
        MERGE (d)-[:MENTIONS]->(e)
        """
        graph.query(cypher, params={"idx": idx, "text": text, "ents": ent_list})


def generate_embeddings_and_store(chunks, graph):
    """Generate embeddings for each chunk and store them as node properties in Neo4j.

    This function uses a local SentenceTransformer model for demonstration.  In
    production you can call Neo4j's genai.vector.encodeBatch procedure or
    integrate with a vector database.  It assumes there is a Document node
    identified by an `id` property matching the index of the chunk.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    for idx, chunk in enumerate(chunks):
        embedding = model.encode(chunk.page_content).tolist()
        cypher = (
            "MATCH (d:Document {id: $idx}) "
            "SET d.embedding = $embedding"
        )
        graph.query(cypher, params={"idx": idx, "embedding": embedding})


def main():
    # Load document(s).  The sample file is located in the `data` folder relative to this script.
    script_dir = os.path.dirname(__file__)
    sample_path = os.path.join(script_dir, 'data', 'sample.txt')
    docs = load_documents(sample_path)
    print(f"Loaded {len(docs)} document(s)")
    # Split into chunks
    chunks = chunk_documents(docs, chunk_size=1024, overlap=128)
    print(f"Generated {len(chunks)} chunks")
    # Initialize spaCy with the small English model.  If you need domain‑specific
    # entities (e.g. drugs), you can load a SciSpacy model instead.  We also
    # add an EntityRuler to recognise custom terms like 'gefapixant'.
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        raise OSError(
            "spaCy model 'en_core_web_sm' not found. Download it with 'python -m spacy download en_core_web_sm'"
        )
    # Add a custom EntityRuler to recognise domain‑specific terms.  spaCy 3
    # requires adding components via their factory names.  We create an
    # EntityRuler through nlp.add_pipe and then add our patterns.
    custom_patterns = [
        {"label": "DRUG", "pattern": "gefapixant"},
        {"label": "ORG", "pattern": "Musa"},  # example additional pattern
    ]
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    ruler.add_patterns(custom_patterns)
    # Connect to Neo4j
    uri = os.environ.get("NEO4J_URI")
    username = os.environ.get("NEO4J_USERNAME")
    password = os.environ.get("NEO4J_PASSWORD")
    if not all([uri, username, password]):
        raise EnvironmentError("NEO4J_URI, NEO4J_USERNAME and NEO4J_PASSWORD must be set")
    graph = Neo4jGraph(url=uri, username=username, password=password, refresh_schema=False)
    # Build graph from chunks
    print("Creating nodes and relationships from chunks...")
    create_graph_from_chunks(chunks, nlp, graph)
    print("Graph creation complete")
    # Generate embeddings and store them
    print("Generating embeddings...")
    generate_embeddings_and_store(chunks, graph)
    print("Embedding generation complete")
    print("Pipeline finished")


if __name__ == '__main__':
    main()
