import json
from pathlib import Path
from typing import Optional
from google import genai
from google.genai import types
from rich.console import Console

from grimoire.config import config, MODEL_ALCHEMY, MODEL_MATERIALIZATION, MODEL_NANO_BANANA
from grimoire.schemas import SigilPrompt
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

def stage_1_alchemy(intent: str, style: str) -> SigilPrompt:
    """
    Stage 1: Alchemy (Gemini 2.5 Pro)
    Synthesizes intent and style into a visual prompt.
    """
    console.print(f"[bold purple]Stage 1: Alchemy (Model: {MODEL_ALCHEMY})[/bold purple]")
    
    client = _get_client(MODEL_ALCHEMY)
    
    system_instruction = """You are a Master Occult Calligrapher and Sigilographer. Your goal is to create a precise visual prompt for generating FUNCTIONAL SIGILS.

    HARD CONSTRAINTS (MUST FOLLOW):
    1. VISUAL OUTPUT MUST BE STRICTLY: "Black ink on white background".
    2. FORBIDDEN TERMS: photorealistic, 3D render, volumetric lighting, colors, shading, depth of field, cinematic, hyperrealistic.
    3. DIMENSIONALITY: The image must be completely FLAT (2D). No shadows, no gradients.
    4. STYLE INTERPRETATION: Treat the 'Style' parameter as a STROKE TECHNIQUE (line-weight, curvature, geometry type), NOT as a thematic subject.
       - Example: If Style is "Temple of Ascending Flame", do NOT draw fire. Instead, use the ARTISTIC SCHOOL of that order (e.g., aggressive geometry, sharp angles, fluid but dangerous curves).
    5. COMPOSITION:
       - User Intent dictates the CENTRAL GEOMETRY.
       - Do NOT hallucinate visual elements that violate the Black & White constraint.
    6. CRITICAL CONSTRAINT: The output prompt MUST be under 480 tokens.
    
    Your output prompt must describe a clean, high-contrast, esoteric diagram suitable for printing or vectorization.
    """
    
    prompt = f"""
    User Intent: {intent}
    Desired Style (Stroke Technique): {style}
    
    Construct the visual prompt.
    Ensure the description enforces a "scanned ancient book diagram" or "clean SVG vector" aesthetic.
    
    Append this exact string to the end of your generated visual prompt:
    "Visual style: High contrast, monochrome, flat 2D, line art, vector graphics aesthetic, esoteric symbol, black ink on white background, no shading, no colors."
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
        logger.error(f"Stage 1 failed: {e}")
        raise ArtificerError(f"Stage 1 (Alchemy) failed: {e}")

def stage_2_materialization(visual_prompt: str, aspect_ratio: str, output_path: Path, use_nano_banana: bool = False):
    """
    Stage 2: Materialization
    Generates the image using either Imagen 4 Ultra or Nano Banana Pro.
    """
    model_name = MODEL_NANO_BANANA if use_nano_banana else MODEL_MATERIALIZATION
    console.print(f"[bold gold1]Stage 2: Materialization (Model: {model_name})[/bold gold1]")
    
    # Use ImagenGuard to get the dedicated paid key for BOTH models to ensure high quality/quota management
    try:
        guard = ImagenGuard()
        api_key = guard.get_key()
        # Create an isolated client for this specific paid operation
        client = genai.Client(api_key=api_key)
    except Exception as e:
        raise ArtificerError(f"Stage 2 Authorization Failed: {e}")
    
    try:
        if use_nano_banana:
            # Nano Banana Pro (Gemini 3 Pro Image Preview) Implementation
            # We use direct REST API call because the installed SDK version (0.2.2) 
            # does not support 'image_config' in GenerateContentConfig yet.
            import requests
            import base64
            
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NANO_BANANA}:generateContent?key={api_key}"
            
            payload = {
                "contents": [{
                    "parts": [{"text": visual_prompt}]
                }],
                "generationConfig": {
                    "responseModalities": ["IMAGE"],
                    "imageConfig": {
                        "aspectRatio": aspect_ratio
                    }
                }
            }
            
            response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
            
            if response.status_code != 200:
                raise ArtificerError(f"Nano Banana Pro API Error ({response.status_code}): {response.text}")
            
            response_data = response.json()
            
            # Parse response for image data
            image_saved = False
            try:
                candidates = response_data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    for part in parts:
                        if "inlineData" in part:
                            b64_data = part["inlineData"]["data"]
                            image_data = base64.b64decode(b64_data)
                            with open(output_path, "wb") as f:
                                f.write(image_data)
                            image_saved = True
                            break
            except Exception as e:
                logger.error(f"Failed to parse Nano Banana response: {e}")
                raise ArtificerError(f"Failed to parse Nano Banana response: {e}")

            if image_saved:
                console.print(f"[green]Sigil saved to {output_path}[/green]")
            else:
                raise ArtificerError("No image generated by Nano Banana Pro (No inlineData found).")

        else:
            # Imagen 4 Ultra Implementation
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
                raise ArtificerError("No image generated by Imagen.")
            
    except Exception as e:
        logger.error(f"Stage 2 failed: {e}")
        raise ArtificerError(f"Stage 2 (Materialization) failed: {e}")

def generate_sigil(intent: str, style: str, aspect_ratio: str, output_path: str, nano_banana_pro: bool = False):
    """
    Orchestrates the Sigil Artificer pipeline.
    """
    output_file = Path(output_path).resolve()
    
    try:
        # --- Stage 1: Alchemy ---
        console.print("[italic]Gemini Pro is distilling your intent into a visual ritual description...[/italic]")
        sigil_prompt = stage_1_alchemy(intent, style)
        console.print(f"[dim]Visual Prompt: {sigil_prompt.visual_prompt}[/dim]")
        
        # --- Stage 2: Materialization ---
        model_display = "Nano Banana Pro" if nano_banana_pro else "Imagen 4 Ultra"
        console.print(f"[italic]Materializing the sigil using the power of {model_display}...[/italic]")
        stage_2_materialization(sigil_prompt.visual_prompt, aspect_ratio, output_file, use_nano_banana=nano_banana_pro)
        
        console.print(f"[bold green]The sigil has been forged and saved to {output_file}.[/bold green]")

    except ArtificerError as e:
        console.print(f"[bold red]The ritual failed:[/bold red] {e}")
    except Exception as e:
        console.print(f"[bold red]An unexpected magical backlash occurred:[/bold red] {e}")
        logger.exception("Unexpected error in generate_sigil")
