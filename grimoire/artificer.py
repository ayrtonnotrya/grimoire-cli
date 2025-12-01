import json
from pathlib import Path
from typing import Optional
from google import genai
from google.genai import types
from rich.console import Console

from grimoire.config import config, MODEL_PLANNING, MODEL_ALCHEMY, MODEL_MATERIALIZATION
from grimoire.schemas import SearchPlan, SigilPrompt
from grimoire import db
from grimoire.keys import key_manager
from grimoire.guard import ImagenGuard
from grimoire.logger import logger

console = Console()

class ArtificerError(Exception):
    """Base exception for Artificer errors."""
    pass

def _get_client(model_name: str) -> genai.Client:
    """Helper to get a client with a valid key."""
    # For now, we use the same key pool. In future, we might want specific keys for specific models if needed.
    # We use reserved_only=False to maximize availability.
    api_key = key_manager.get_best_key(reserved_only=False)
    if not api_key:
        raise ArtificerError("All API Keys exhausted.")
    return genai.Client(api_key=api_key)

def stage_1_planning(intent: str, style: str) -> SearchPlan:
    """
    Stage 1: Planning (Gemini 2.5 Flash-Lite)
    Generates search queries based on intent and style.
    """
    console.print(f"[bold blue]Stage 1: Planning (Model: {MODEL_PLANNING})[/bold blue]")
    
    client = _get_client(MODEL_PLANNING)
    
    system_instruction = """You are an expert occult librarian. Your goal is to formulate precise search queries to retrieve relevant magical knowledge from a vector database.
    The user wants to create a 'Sigil' (magical symbol) for a specific intent and style.
    
    Constraints:
    1. The next stage has a context limit. Do NOT retrieve the entire library.
    2. Focus on the specific intent and style.
    3. Return ONLY the JSON object matching the SearchPlan schema.
    """
    
    prompt = f"""
    User Intent: {intent}
    Desired Style: {style}
    
    Generate 3-5 precise search queries to find relevant symbols, rituals, and artistic references in the occult library.
    """
    
    try:
        response = client.models.generate_content(
            model=MODEL_PLANNING,
            contents=prompt,
            config={
                'response_mime_type': 'application/json',
                'response_schema': SearchPlan,
                'system_instruction': system_instruction
            }
        )
        
        plan = SearchPlan(**json.loads(response.text))
        return plan
    except Exception as e:
        logger.error(f"Stage 1 failed: {e}")
        raise ArtificerError(f"Stage 1 (Planning) failed: {e}")

def stage_2_alchemy(intent: str, style: str, search_results: list[str]) -> SigilPrompt:
    """
    Stage 2: Alchemy (Gemini 2.5 Pro)
    Synthesizes retrieved knowledge into a visual prompt.
    """
    console.print(f"[bold purple]Stage 2: Alchemy (Model: {MODEL_ALCHEMY})[/bold purple]")
    
    client = _get_client(MODEL_ALCHEMY)
    
    context_text = "\n\n".join(search_results)
    
    system_instruction = """You are a Sigil Artificer, a master of visual magic. Your goal is to create a DENSE, DESCRIPTIVE visual prompt for an AI image generator (Imagen 4 Ultra).
    
    CRITICAL CONSTRAINT: The output prompt MUST be under 480 tokens.
    
    Instructions:
    1. Analyze the User Intent, Style, and the Retrieved Occult Context.
    2. Synthesize a visual description of a sigil that embodies these elements.
    3. Focus on: Geometry, Line weight, Colors (or lack thereof), Texture, Lighting, and Atmosphere.
    4. Do NOT include conversational filler ("Here is the prompt"). Return ONLY the JSON.
    """
    
    prompt = f"""
    User Intent: {intent}
    Desired Style: {style}
    
    Retrieved Context from Grimoire:
    {context_text}
    
    Create the visual prompt now.
    """
    
    try:
        response = client.models.generate_content(
            model=MODEL_ALCHEMY,
            contents=prompt,
            config={
                'response_mime_type': 'application/json',
                'response_schema': SigilPrompt,
                'system_instruction': system_instruction
            }
        )
        
        sigil_prompt = SigilPrompt(**json.loads(response.text))
        return sigil_prompt
    except Exception as e:
        logger.error(f"Stage 2 failed: {e}")
        raise ArtificerError(f"Stage 2 (Alchemy) failed: {e}")

def stage_3_materialization(visual_prompt: str, aspect_ratio: str, output_path: Path):
    """
    Stage 3: Materialization (Imagen 4 Ultra)
    Generates the image.
    """
    console.print(f"[bold gold1]Stage 3: Materialization (Model: {MODEL_MATERIALIZATION})[/bold gold1]")
    
    # Use ImagenGuard to get the dedicated paid key
    try:
        guard = ImagenGuard()
        api_key = guard.get_key()
        # Create an isolated client for this specific paid operation
        client = genai.Client(api_key=api_key)
    except Exception as e:
        raise ArtificerError(f"Stage 3 Authorization Failed: {e}")
    
    try:
        # Imagen 4 Ultra parameters
        # Note: The python client might have slightly different parameter names depending on version.
        # Based on docs/genai/generate_images_using_Imagen.md, we use 'imagen-3.0-generate-001' style but for 4.0.
        # Assuming standard generate_images call.
        
        response = client.models.generate_image(
            model=MODEL_MATERIALIZATION,
            prompt=visual_prompt,
            config=types.GenerateImageConfig(
                number_of_images=1,
                aspect_ratio=aspect_ratio,
                person_generation='DONT_ALLOW',
                safety_filter_level="BLOCK_LOW_AND_ABOVE", 
            )
        )
        
        if response.generated_images:
            image = response.generated_images[0]
            image.image.save(output_path)
            console.print(f"[green]Sigil saved to {output_path}[/green]")
        else:
            raise ArtificerError("No image generated.")
            
    except Exception as e:
        logger.error(f"Stage 3 failed: {e}")
        raise ArtificerError(f"Stage 3 (Materialization) failed: {e}")

def generate_sigil(intent: str, style: str, aspect_ratio: str, output_path: str):
    """
    Orchestrates the Sigil Artificer pipeline.
    """
    output_file = Path(output_path).resolve()
    
    try:
        # --- Stage 1: Planning ---
        console.print("[italic]Summoning Gemini Flash-Lite to interpret your intent and consult the ethereal archives...[/italic]")
        plan = stage_1_planning(intent, style)
        console.print(f"[cyan]Queries generated:[/cyan] {plan.search_queries}")
        
        # --- Intermediate: Retrieval ---
        console.print(f"[italic]Consulting the library...[/italic]")
        all_results = []
        for query in plan.search_queries:
            # Increased to 100 results per query to leverage Gemini 2.5 Pro's 2M+ token window
            results = db.query_documents(query, n_results=100)
            if results['documents'] and results['documents'][0]:
                all_results.extend(results['documents'][0])
        
        if not all_results:
            console.print("[yellow]Warning: No arcane references found. Proceeding with raw intent.[/yellow]")
        else:
            console.print(f"[green]Found {len(all_results)} arcane references.[/green]")

        # --- Stage 2: Alchemy ---
        console.print("[italic]Gemini Pro is now weaving the retrieved concepts into a visual ritual description (Targeting < 480 tokens)...[/italic]")
        sigil_prompt = stage_2_alchemy(intent, style, all_results)
        console.print(f"[dim]Visual Prompt: {sigil_prompt.visual_prompt}[/dim]")
        
        # --- Stage 3: Materialization ---
        console.print("[italic]Materializing the sigil using the power of Imagen 4 Ultra...[/italic]")
        stage_3_materialization(sigil_prompt.visual_prompt, aspect_ratio, output_file)
        
        console.print(f"[bold green]The sigil has been forged and saved to {output_file}.[/bold green]")

    except ArtificerError as e:
        console.print(f"[bold red]The ritual failed:[/bold red] {e}")
    except Exception as e:
        console.print(f"[bold red]An unexpected magical backlash occurred:[/bold red] {e}")
        logger.exception("Unexpected error in generate_sigil")
