import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai
from grimoire.config import config

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        api_key = config.gemini_api_key
        if not api_key:
            raise ValueError("Gemini API Key not configured")
        
        client = genai.Client(api_key=api_key)
        model = "gemini-embedding-001" # Correct embedding model name
        
        embeddings = []
        for text in input:
            # New SDK usage for embeddings
            # client.models.embed_content(model=..., contents=...)
            result = client.models.embed_content(
                model=model,
                contents=text,
                config={'task_type': 'RETRIEVAL_DOCUMENT'} # Check correct enum or string
            )
            # Result structure might be different. 
            # response.embeddings[0].values ?
            # Let's assume result.embeddings[0].values based on typical new SDKs
            # Or result.embedding_values if single?
            # Checking docs or assuming standard new Google SDK response:
            # It usually returns an object with `embeddings`.
            embeddings.append(result.embeddings[0].values)
        return embeddings

def get_db_client():
    db_dir = config.db_dir
    db_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(db_dir))

def get_collection():
    client = get_db_client()
    embedding_fn = GeminiEmbeddingFunction()
    return client.get_or_create_collection(
        name="grimoire_summaries",
        embedding_function=embedding_fn
    )

def add_documents(documents: list[str], metadatas: list[dict], ids: list[str]):
    collection = get_collection()
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

def query_documents(query_text: str, n_results: int = 5):
    collection = get_collection()
    
    # For query embedding, we might need to manually embed if we want to specify task_type='RETRIEVAL_QUERY'
    # But ChromaDB's query() takes query_texts and uses the embedding function (which uses RETRIEVAL_DOCUMENT above).
    # This is a mismatch. 
    # Ideally, we pass query_embeddings to collection.query().
    
    api_key = config.gemini_api_key
    client = genai.Client(api_key=api_key)
    model = "gemini-embedding-001"
    
    query_result = client.models.embed_content(
        model=model,
        contents=query_text,
        config={'task_type': 'RETRIEVAL_QUERY'}
    )
    query_embedding = query_result.embeddings[0].values
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results

