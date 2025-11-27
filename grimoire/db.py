import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from google import genai
from google.genai import errors
from grimoire.config import config
import time
import threading
import random
from grimoire.logger import logger

DB_LOCK = threading.Lock()
_CLIENT = None
_CLIENT_LOCK = threading.Lock()

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        api_key = config.gemini_api_key
        if not api_key:
            raise ValueError("Gemini API Key not configured")
        
        client = genai.Client(api_key=api_key)
        model = "gemini-embedding-001" # Correct embedding model name
        
        # Process in batches of 100 to respect rate limits (100 requests per minute)
        batch_size = 99
        delay_seconds = 61  
        
        embeddings = []
        total_texts = len(input)
        
        for batch_start in range(0, total_texts, batch_size):
            batch_end = min(batch_start + batch_size, total_texts)
            batch = input[batch_start:batch_end]
            
            # Print progress if processing multiple batches
            if total_texts > batch_size:
                print(f"Processing batch {batch_start // batch_size + 1}/{(total_texts + batch_size - 1) // batch_size} ({batch_start + 1}-{batch_end} of {total_texts})")
            
            # Process current batch
            for text in batch:
                result = client.models.embed_content(
                    model=model,
                    contents=text,
                    config={'task_type': 'RETRIEVAL_DOCUMENT'}
                )
                embeddings.append(result.embeddings[0].values)
            
            # Add delay between batches (except after the last batch)
            if batch_end < total_texts:
                print(f"Waiting {delay_seconds} seconds before next batch to respect rate limits...")
                time.sleep(delay_seconds)
        
        return embeddings

def generate_embeddings(texts: list[str], api_key: str) -> list[list[float]]:
    """Generates embeddings for a list of texts using a specific API key."""
    if api_key in config.exhausted_keys:
        raise RuntimeError("API Key exhausted (Daily Limit)")

    client = genai.Client(api_key=api_key)
    model = "gemini-embedding-001"
    
    # Process in batches of 100 to respect rate limits
    # Process in batches of 5 to respect rate limits (30k TPM)
    # 5 chunks * ~1000 chars/chunk ~= 1-2k tokens per batch.
    # With 4 threads, max 8k tokens if all hit at once.
    # 100 RPM limit is safe (1 request per 5 chunks).
    batch_size = 5
    delay_seconds = 2 # Small delay just to be nice
    
    embeddings = []
    total_texts = len(texts)
    
    for batch_start in range(0, total_texts, batch_size):
        batch_end = min(batch_start + batch_size, total_texts)
        batch = texts[batch_start:batch_end]
        
        # Process current batch with retry logic
        max_retries = 5
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Batch embedding call
                result = client.models.embed_content(
                    model=model,
                    contents=batch,
                    config={'task_type': 'RETRIEVAL_DOCUMENT'}
                )
                for embedding in result.embeddings:
                    embeddings.append(embedding.values)
                break # Success, exit retry loop
            except (errors.ClientError, errors.ServerError) as e:
                # Check for 429 (Too Many Requests) or 503 (Service Unavailable)
                if e.code not in [429, 503]:
                    raise e
                
                if attempt == max_retries - 1:
                    masked_key = f"...{api_key[-4:]}"
                    
                    # Check for Daily Limit
                    if "RequestsPerDay" in str(e) or "Daily" in str(e):
                        config.exhausted_keys.add(api_key)
                        raise RuntimeError(f"Daily Rate Limit Exceeded (Key: {masked_key})") from e
                    
                    raise RuntimeError(f"{e} (Key: {masked_key})") from e
                
                # Exponential backoff with jitter
                delay = (base_delay * (2 ** attempt)) + (random.random() * 0.5)
                logger.warning(f"Rate limit hit for key ...{api_key[-4:]}. Retrying in {delay:.2f}s... Error: {e}")
                time.sleep(delay)
                
                # If this was the last attempt, we want to make sure the error message includes the key
                # But we re-raise above. Let's wrap the re-raise to add info.
        
        # Add delay between batches
        if batch_end < total_texts:
            time.sleep(delay_seconds)
            
    return embeddings

def get_db_client():
    global _CLIENT
    if _CLIENT is None:
        with _CLIENT_LOCK:
            if _CLIENT is None:
                db_dir = config.db_dir
                db_dir.mkdir(parents=True, exist_ok=True)
                _CLIENT = chromadb.PersistentClient(path=str(db_dir))
    return _CLIENT

def get_collection():
    client = get_db_client()
    embedding_fn = GeminiEmbeddingFunction()
    return client.get_or_create_collection(
        name="grimoire_summaries",
        embedding_function=embedding_fn
    )

def add_documents(documents: list[str], metadatas: list[dict], ids: list[str], embeddings: list[list[float]] = None, verbose: bool = False):
    collection = get_collection()
    
    with DB_LOCK:
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )

def query_documents(query_text: str, n_results: int = 5):
    collection = get_collection()
    
    # For query embedding, we might need to manually embed if we want to specify task_type='RETRIEVAL_QUERY'
    # But ChromaDB's query() takes query_texts and uses the embedding function (which uses RETRIEVAL_DOCUMENT above).
    # This is a mismatch. 
    # Ideally, we pass query_embeddings to collection.query().
    
    api_keys = config.gemini_api_keys
    last_error = None

    for api_key in api_keys:
        try:
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
        except Exception as e:
            last_error = e
            masked_key = f"...{api_key[-4:]}"
            logger.warning(f"Search failed with key {masked_key}: {e}. Rotating key...")
            continue
            
    # If we get here, all keys failed
    raise RuntimeError(f"All API keys failed. Last error: {last_error}")

def document_exists(filename: str) -> bool:
    """Checks if a document with the given filename exists in the collection."""
    collection = get_collection()
    # We only need to check if any document exists with this filename in metadata
    results = collection.get(
        where={"filename": filename},
        limit=1
    )
    return len(results['ids']) > 0

def get_document_path(filename: str) -> str | None:
    """Retrieves the full_path from metadata for a given filename."""
    collection = get_collection()
    results = collection.get(
        where={"filename": filename},
        limit=1,
        include=["metadatas"]
    )
    if results['metadatas'] and results['metadatas'][0]:
        return results['metadatas'][0].get('full_path')
    return None

def update_document_path(filename: str, new_path: str):
    """Updates the full_path metadata for all chunks associated with the given filename."""
    collection = get_collection()
    
    # Get all documents for this file
    results = collection.get(
        where={"filename": filename},
        include=["metadatas"]
    )
    
    if not results['ids']:
        return

    ids_to_update = results['ids']
    metadatas_to_update = []
    
    for metadata in results['metadatas']:
        # Create a copy and update the path
        new_metadata = metadata.copy()
        new_metadata['full_path'] = new_path
        metadatas_to_update.append(new_metadata)
        
    collection.update(
        ids=ids_to_update,
        metadatas=metadatas_to_update
    )


def remove_duplicates() -> int:
    """Removes duplicate documents from the collection based on content hash."""
    collection = get_collection()
    all_docs = collection.get()
    
    if not all_docs['ids']:
        return 0
        
    # Map (content, chunk_type, title) -> list of IDs
    content_map = {}
    
    for i, doc_id in enumerate(all_docs['ids']):
        content = all_docs['documents'][i]
        metadata = all_docs['metadatas'][i]
        
        # Create a unique signature for the content
        # We use content + chunk_type + title to be safe
        key = (
            content,
            metadata.get('chunk_type', ''),
            metadata.get('title', '')
        )
        
        if key not in content_map:
            content_map[key] = []
        content_map[key].append(doc_id)
        
    ids_to_delete = []
    
    for key, ids in content_map.items():
        if len(ids) > 1:
            # Sort IDs to ensure deterministic behavior (keep the first one)
            # You might want to keep the one that matches the current filename format if possible,
            # but simple sorting is usually enough for pure duplicates.
            ids.sort()
            
            # Keep the first one, delete the rest
            ids_to_delete.extend(ids[1:])
            
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        
    return len(ids_to_delete)

