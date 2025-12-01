import json
from google import genai
from google.genai import types
from grimoire.config import config
from grimoire.keys import key_manager
from grimoire.db import query_documents
from grimoire.schemas import SearchPlan, ConstructedRitual, CONSTRUCTED_RITUAL_SCHEMA
from grimoire.logger import logger

def perform_ritual_planning(intent: str, inventory: str) -> ConstructedRitual:
    """
    Orchestrates the ritual creation process in two stages:
    1. The Librarian: Translates intent into search queries.
    2. The Master Ritualist: Constructs the ritual using context and inventory.
    """
    
    # --- Stage 1: The Librarian (Gemini 2.5 Flash-Lite) ---
    logger.info("Stage 1: The Librarian is researching...")
    
    librarian_model = "gemini-2.5-flash-lite"
    
    librarian_prompt = f"""
    You are The Librarian, an expert in occult knowledge retrieval.
    Your task is to translate the user's vague intent into precise search queries to find relevant rituals, correspondences, and magical theory in our library.
    
    User Intent: "{intent}"
    
    Generate 3 to 5 distinct search queries that cover:
    - Specific rituals related to the intent.
    - Deities, spirits, or forces associated with the intent.
    - Correspondences (herbs, stones, colors, planetary hours).
    - Theoretical underpinnings (e.g., "banishing", "invocation", "evocation").
    """
    
    # Acquire key and rate limit
    api_key = key_manager.get_best_key()
    if not api_key:
        raise RuntimeError("No API keys available.")
    
    key_manager.acquire(api_key, estimated_tokens=500)
    client = genai.Client(api_key=api_key)
    
    try:
        response = client.models.generate_content(
            model=librarian_model,
            contents=librarian_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SearchPlan
            )
        )
        search_plan = response.parsed
    except Exception as e:
        logger.error(f"The Librarian failed: {e}")
        raise RuntimeError(f"The Librarian failed to generate a search plan: {e}")

    # Execute Search
    context_fragments = []
    for query in search_plan.search_queries:
        logger.info(f"Searching for: {query}")
        try:
            results = query_documents(query, n_results=3)
            if results['documents'] and results['documents'][0]:
                context_fragments.extend(results['documents'][0])
        except Exception as e:
            logger.warning(f"Search failed for query '{query}': {e}")
            
    full_context = "\n\n".join(context_fragments)
    
    # --- Stage 2: The Master Ritualist (Gemini 2.5 Pro) ---
    logger.info("Stage 2: The Master Ritualist is constructing the ritual...")
    
    ritualist_model = "gemini-2.5-pro"
    
    ritualist_prompt = f"""
    You are The Master Ritualist. You have deep knowledge of the occult and a practical, pragmatic approach to magic.
    Your task is to construct a complete, workable ritual for the user based on their intent, their available inventory, and the knowledge retrieved from the library.
    
    User Intent: "{intent}"
    User Inventory: "{inventory}"
    
    Library Knowledge (Context):
    {full_context}
    
    Instructions:
    1. Synthesize the Library Knowledge to ensure the ritual is grounded in tradition and theory.
    2. ADAPT aggressively to the User Inventory. If a traditional tool is missing, use your creativity to substitute it with something from the inventory (e.g., use a kitchen knife instead of an Athame, salt instead of chalk).
    3. If the inventory is extremely sparse, design a "minimalist" version of the ritual (e.g., using only visualization, breath, or body posture).
    4. Structure the ritual clearly with a Title, Timing, Tools, Steps, Closing, and Expected Result.
    5. The 'Steps' should include physical actions, verbal incantations (write them out!), and mental visualizations.
    """
    
    # Acquire key and rate limit (might need a fresh key check)
    api_key = key_manager.get_best_key() 
    if not api_key:
        raise RuntimeError("No API keys available.")

    key_manager.acquire(api_key, estimated_tokens=2000)
    client = genai.Client(api_key=api_key)
    
    try:
        response = client.models.generate_content(
            model=ritualist_model,
            contents=ritualist_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CONSTRUCTED_RITUAL_SCHEMA
            )
        )
        
        # Manually parse JSON response into Pydantic model
        # The response.parsed might be a dict or object depending on SDK version when using dict schema
        # Usually response.text contains the JSON string.
        # Let's try to parse response.text
        import json
        data = json.loads(response.text)
        return ConstructedRitual(**data)
        
    except Exception as e:
        logger.error(f"The Master Ritualist failed: {e}")
        raise RuntimeError(f"The Master Ritualist failed to construct the ritual: {e}")
