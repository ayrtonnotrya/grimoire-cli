import time
import threading
from collections import deque
import os
import concurrent.futures
import subprocess
import shutil
from pathlib import Path
from typing import Optional
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.console import Console
from grimoire.logger import logger
from google import genai
from google.genai import types
from grimoire.config import config
from grimoire.schemas import BookSummary, BOOK_SUMMARY_SCHEMA
import json
import random
from enum import Enum, auto

console = Console()

class Action(Enum):
    RETRY = auto()
    ROTATE_KEY = auto()
    SKIP_FILE = auto()
    ABORT = auto()

from grimoire.keys import key_manager

def get_latest_library_tree(directory: Path) -> Path:
    """Finds the latest library_tree_*.txt file in the directory."""
    files = list(directory.glob("library_tree_*.txt"))
    if not files:
        raise FileNotFoundError(f"No library_tree_*.txt found in {directory}")
    return max(files, key=os.path.getctime)

def parse_library_list(file_path: Path) -> list[Path]:
    """Parses the library list file and returns a list of PDF paths.
    
    Expected format: One absolute PDF path per line.
    Example:
        /path/to/book1.pdf
        /path/to/book2.pdf
    """
    pdf_paths = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            
            # Check if line ends with .pdf
            if line.endswith(".pdf"):
                pdf_path = Path(line)
                pdf_paths.append(pdf_path)
    
    return pdf_paths

def check_summary_exists(pdf_name: str) -> bool:
    """Checks if a summary file already exists for the given PDF."""
    summary_path = config.summaries_dir / f"summary_{pdf_name}.json"
    return summary_path.exists()

def process_single_file(pdf_path: Path, verbose: bool = False) -> dict:
    """Processes a single PDF file."""
    api_keys = config.gemini_api_keys
    if not api_keys:
        return {"status": "error", "file": pdf_path.name, "message": "No API keys configured"}
    
    if check_summary_exists(pdf_path.name):
        return {"status": "skipped", "file": pdf_path.name, "message": "Summary already exists"}

    return generate_summary(pdf_path, api_keys)

def process_library(list_file_path: str, exclude_file_path: str = None, sequential: bool = False, verbose: bool = False):
    """Main processing function."""
    file_path = Path(list_file_path)
    
    # If directory is passed, find latest tree file
    if file_path.is_dir():
        try:
            file_path = get_latest_library_tree(file_path)
            console.print(f"[blue]Using latest library tree: {file_path}[/blue]")
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            return

    console.print(f"[bold]Parsing {file_path}...[/bold]")
    pdf_paths = parse_library_list(file_path)
    console.print(f"Found {len(pdf_paths)} PDFs.")

    if not sequential:
        console.print("[yellow]Shuffling processing order...[/yellow]")
        random.shuffle(pdf_paths)
    else:
        console.print("[blue]Processing in sequential order.[/blue]")

    api_keys = config.gemini_api_keys
    if not api_keys:
        console.print("[red]Error: Gemini API Key not configured. Run 'grimoire init'.[/red]")
        return

    # Filter out PDFs that already have summaries
    pdfs_to_process = []
    skipped_count = 0
    
    # Load exclusion list if provided
    excluded_paths = set()
    if exclude_file_path:
        try:
            console.print(f"[bold]Loading exclusion list from {exclude_file_path}...[/bold]")
            # Use parse_library_list to reuse the logic (stripping, ignoring comments)
            excluded_list = parse_library_list(Path(exclude_file_path))
            # Convert to absolute strings for comparison
            excluded_paths = {str(p.resolve()) for p in excluded_list}
            console.print(f"[yellow]Loaded {len(excluded_paths)} files to exclude.[/yellow]")
        except Exception as e:
            console.print(f"[red]Failed to load exclusion list: {e}[/red]")
            return

    for pdf_path in pdf_paths:
        # Check if file is in exclusion list
        if str(pdf_path.resolve()) in excluded_paths:
            if verbose:
                console.print(f"[yellow]Skipping excluded file: {pdf_path.name}[/yellow]")
            continue

        if check_summary_exists(pdf_path.name):
             skipped_count += 1
             if verbose:
                console.print(f"[yellow]Skipping existing summary for: {pdf_path.name}[/yellow]")
             continue
        pdfs_to_process.append(pdf_path)

    if skipped_count > 0:
        console.print(f"[yellow]Skipped {skipped_count} existing summaries.[/yellow]")

    if not pdfs_to_process:
        console.print("[green]No new books to process.[/green]")
        return

    console.print(f"[bold blue]Processing {len(pdfs_to_process)} books...[/bold blue]")

    # Define error log file for 400 errors
    error_log_path = file_path.parent / "process_failures_400.txt"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task_id = progress.add_task("Summarizing...", total=len(pdfs_to_process))
        
        # Parallel Processing
        if len(api_keys) > 1:
            # Use manual executor management to avoid hanging on shutdown
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(api_keys))
            future_to_path = {}
            try:
                for i, pdf_path in enumerate(pdfs_to_process):
                    # Pass ALL keys to the worker, let it manage retries
                    future = executor.submit(generate_summary, pdf_path, api_keys)
                    future_to_path[future] = pdf_path
                
                for future in concurrent.futures.as_completed(future_to_path):
                    pdf_path = future_to_path[future]
                    try:
                        result = future.result()
                        progress.advance(task_id)
                        _print_process_result(progress.console, result, verbose=verbose)
                        
                        # Log 400 errors
                        if result["status"] == "error" and "400" in result.get("message", ""):
                            try:
                                with open(error_log_path, "a") as f:
                                    f.write(f"{pdf_path}\n")
                            except Exception as e:
                                logger.error(f"Failed to log 400 error: {e}")

                        # Check for global exhaustion
                        if result["status"] == "skipped" and "All API Keys exhausted" in result["message"]:
                            progress.console.print("[bold red]CRITICAL: All API Keys have been exhausted (Daily Limits). Stopping process.[/bold red]")
                            progress.console.print(f"[red]Please check the logs for details: {config.log_file}[/red]")
                            # Cancel all other futures and shutdown immediately
                            executor.shutdown(wait=False, cancel_futures=True)
                            return # Exit function immediately
                    except concurrent.futures.CancelledError:
                        continue
            finally:
                # Ensure executor is shut down
                executor.shutdown(wait=False)
        else:
            # Sequential Processing
            for pdf_path in pdfs_to_process:
                # Check exhaustion before starting next
                if len(config.exhausted_keys) == len(api_keys):
                     progress.console.print("[bold red]CRITICAL: All API Keys have been exhausted (Daily Limits). Stopping process.[/bold red]")
                     progress.console.print(f"[red]Please check the logs for details: {config.log_file}[/red]")
                     break

                result = generate_summary(pdf_path, api_keys)
                progress.advance(task_id)
                _print_process_result(progress.console, result, verbose=verbose)

                # Log 400 errors
                if result["status"] == "error" and "400" in result.get("message", ""):
                    try:
                        with open(error_log_path, "a") as f:
                            f.write(f"{pdf_path}\n")
                    except Exception as e:
                        logger.error(f"Failed to log 400 error: {e}")
                
                
                if result["status"] == "skipped" and "All API Keys exhausted" in result["message"]:
                     progress.console.print("[bold red]CRITICAL: All API Keys have been exhausted (Daily Limits). Stopping process.[/bold red]")
                     progress.console.print(f"[red]Please check the logs for details: {config.log_file}[/red]")
                     break

def _print_process_result(console, result, verbose: bool = False):
    color = "green" if result["status"] == "success" else "red"
    icon = "✓" if result["status"] == "success" else "✗"
    
    msg = f"[{color}]{icon} {result['file']}[/{color}]"
    if result.get("masked_key"):
        msg += f" [dim]({result['masked_key']})[/dim]"
    msg += f"\n[dim]{result['message']}[/dim]"
    
    if verbose and result.get("full_error"):
        msg += f"\n[dim red]Full Error: {result['full_error']}[/dim red]"

    console.print(
        Panel(
            msg,
            border_style=color
        )
    )

def handle_api_error(error: Exception, api_key: str, file_name: str) -> tuple[Action, str]:
    """Analyzes the API error and returns the appropriate action and user message."""
    error_str = str(error)
    masked_key = f"...{api_key[-4:]}"
    
    # Special Case: Daily Rate Limit Exceeded (Can appear without 429)
    if "Daily Rate Limit Exceeded" in error_str:
        return Action.ROTATE_KEY, f"Cota diária excedida (Daily Rate Limit Exceeded). Chave {masked_key} marcada como inválida."

    # 400 INVALID_ARGUMENT
    if "400" in error_str and "INVALID_ARGUMENT" in error_str:
        return Action.SKIP_FILE, f"Erro na solicitação (400). Verifique o formato do arquivo ou prompt. (Arquivo: {file_name})"

    # 400 FAILED_PRECONDITION (Free tier not supported/Billing inactive)
    if "400" in error_str and "FAILED_PRECONDITION" in error_str:
        return Action.ROTATE_KEY, f"Nível gratuito não suportado ou faturamento inativo (400). Chave {masked_key} marcada como inválida."

    # 403 PERMISSION_DENIED (Key suspended, wrong key, etc.)
    if "403" in error_str or "PERMISSION_DENIED" in error_str or "CONSUMER_SUSPENDED" in error_str:
        return Action.ROTATE_KEY, f"Permissão negada/Chave suspensa (403). Chave {masked_key} marcada como inválida."

    # 404 NOT_FOUND
    if "404" in error_str or "NOT_FOUND" in error_str:
        return Action.SKIP_FILE, f"Recurso não encontrado (404). (Arquivo: {file_name})"

    # 429 RESOURCE_EXHAUSTED (Daily quota or rate limit)
    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
        # Check for specific "Quota exceeded" message
        if "Quota exceeded" in error_str or "Daily" in error_str:
             return Action.ROTATE_KEY, f"Cota diária excedida (Quota exceeded). Chave {masked_key} marcada como inválida."

        # Transient Rate Limit (RPM/TPM)
        # Do NOT rotate key immediately. Retry with backoff.
        return Action.RETRY, f"Limite de taxa (429). Aguardando para tentar novamente com chave {masked_key}..."

    # 500 INTERNAL
    if "500" in error_str or "INTERNAL" in error_str:
        return Action.RETRY, "Erro interno do Google (500). Tentando novamente..."

    # 503 UNAVAILABLE
    if "503" in error_str or "UNAVAILABLE" in error_str:
        return Action.RETRY, "Serviço indisponível (503). Tentando novamente..."

    # 504 DEADLINE_EXCEEDED
    if "504" in error_str or "DEADLINE_EXCEEDED" in error_str:
        return Action.RETRY, "Tempo limite excedido (504). Tentando novamente..."

    # Default fallback
    return Action.ABORT, f"Erro desconhecido: {error_str}"

def generate_summary(pdf_path: Path, api_keys: list[str]) -> dict:
    """Generates a summary for the given PDF using Gemini, with retry and rotation logic."""
    import random
    
    # Filter out exhausted keys
    available_keys = [k for k in api_keys if k not in config.exhausted_keys]
    
    if not available_keys:
        return {"status": "skipped", "file": pdf_path.name, "message": "All API Keys exhausted (Daily Limit)"}

    # Load prompt once
    current_dir = Path(__file__).parent
    prompt_path = current_dir / "templates" / "book_summary_prompt.md"
    if not prompt_path.exists():
         return {"status": "error", "file": pdf_path.name, "message": f"Prompt file not found at {prompt_path}"}
    with open(prompt_path, "r") as f:
        prompt_text = f.read()

    try:
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()
    except Exception as e:
        return {"status": "error", "file": pdf_path.name, "message": f"Failed to read file: {e}"}

    # Retry loop
    max_retries = 3
    retry_count = 0
    
    while available_keys:
        # Pick a key
        api_key = key_manager.get_best_key()
        if not api_key:
             return {"status": "skipped", "file": pdf_path.name, "message": "All API Keys exhausted (Daily Limit)"}
        client = genai.Client(api_key=api_key)
        
        try:
            # Count tokens (Best effort)
            try:
                token_count_resp = client.models.count_tokens(
                    model=config.model_name,
                    contents=[
                        types.Part.from_bytes(data=pdf_data, mime_type='application/pdf'),
                        prompt_text
                    ]
                )
                estimated_tokens = token_count_resp.total_tokens
            except Exception:
                # Fallback estimation
                estimated_tokens = int(pdf_path.stat().st_size / 4)

            # Acquire rate limit
            key_manager.acquire(api_key, estimated_tokens)

            # Generate content
            response = client.models.generate_content(
                model=config.model_name,
                contents=[
                    types.Part.from_bytes(data=pdf_data, mime_type='application/pdf'),
                    prompt_text
                ],
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': BOOK_SUMMARY_SCHEMA,
                    'safety_settings': [
                        types.SafetySetting(category='HARM_CATEGORY_HATE_SPEECH', threshold='BLOCK_NONE'),
                        types.SafetySetting(category='HARM_CATEGORY_DANGEROUS_CONTENT', threshold='BLOCK_NONE'),
                        types.SafetySetting(category='HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold='BLOCK_NONE'),
                        types.SafetySetting(category='HARM_CATEGORY_HARASSMENT', threshold='BLOCK_NONE'),
                    ]
                }
            )
            
            # Parse and Save
            summary_dict = json.loads(response.text)
            summary_data = BookSummary(**summary_dict)
            
            json_path = config.summaries_dir / f"summary_{pdf_path.name}.json"
            config.summaries_dir.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w") as f:
                f.write(summary_data.model_dump_json(indent=2))
                
            return {"status": "success", "file": pdf_path.name, "message": f"Saved to {json_path.name}", "masked_key": f"...{api_key[-4:]}"}

        except Exception as e:
            action, msg = handle_api_error(e, api_key, pdf_path.name)
            
            # Enhanced Logging for API Errors
            error_details = str(e)
            # Try to extract full response from ClientError
            if hasattr(e, 'response'):
                 try:
                     # If it's a ClientError or similar with a response object
                     if hasattr(e.response, 'text'):
                         error_details += f"\nAPI Response JSON: {e.response.text}"
                     elif hasattr(e.response, 'json'):
                         error_details += f"\nAPI Response JSON: {json.dumps(e.response.json(), indent=2)}"
                 except Exception:
                     pass

            logger.error(f"Full error for {pdf_path.name}: {error_details}", exc_info=True)
            logger.warning(f"Error processing {pdf_path.name}: {msg}")
            
            if action == Action.ROTATE_KEY:
                config.exhausted_keys.add(api_key)
                if api_key in available_keys:
                    available_keys.remove(api_key)
                logger.error(f"{msg} - Rotating key.")
                continue # Try next key immediately

            elif action == Action.RETRY:
                retry_count += 1
                if retry_count <= max_retries:
                    # Increase backoff for 429s
                    backoff = 2 * retry_count
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        backoff = 10 * retry_count # Aggressive backoff for rate limits (10s, 20s, 30s)
                    
                    logger.info(f"{msg} - Retrying ({retry_count}/{max_retries}) in {backoff}s...")
                    time.sleep(backoff)
                    continue # Retry with same key
                else:
                    logger.error(f"Max retries exceeded for {pdf_path.name}.")
                    # If max retries exceeded for 429, it might be a quota issue, BUT we shouldn't kill the key
                    # unless we are sure. Just fail this file to avoid killing the run.
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                         logger.warning(f"Max retries for 429 exceeded. Skipping file {pdf_path.name} but keeping key {api_key[-4:]} alive.")
                         return {"status": "error", "file": pdf_path.name, "message": f"Rate limit persistent: {msg}", "full_error": str(e), "masked_key": f"...{api_key[-4:]}"}

                    return {"status": "error", "file": pdf_path.name, "message": f"Max retries exceeded: {msg}", "full_error": str(e), "masked_key": f"...{api_key[-4:]}"}

            elif action == Action.SKIP_FILE:
                logger.error(f"Skipping file {pdf_path.name}: {msg}")
                return {"status": "error", "file": pdf_path.name, "message": msg}
            
            else: # ABORT or unknown
                return {"status": "error", "file": pdf_path.name, "message": msg}

    return {"status": "skipped", "file": pdf_path.name, "message": "All API Keys exhausted (Daily Limit)"}


def index_single_file(summary_path: Path, verbose: bool = False) -> dict:
    """Indexes a single summary file."""
    api_keys = config.gemini_api_keys
    if not api_keys:
        return {"status": "error", "file": summary_path.name, "message": "No API keys configured"}
    
    return _index_single_book(summary_path, api_keys)

def index_summaries(verbose: bool = False):
    """Reads all summary files and indexes them in ChromaDB."""
    from grimoire import db
    summaries_dir = config.summaries_dir
    if not summaries_dir.exists():
        console.print(f"[yellow]No summaries directory found at {summaries_dir}[/yellow]")
        return

    # Look for JSON files now
    files = list(summaries_dir.glob("summary_*.json"))
    if not files:
        console.print("[yellow]No JSON summaries found to index.[/yellow]")
        return

    documents = []
    metadatas = []
    ids = []

    console.print(f"Found {len(files)} summaries. Checking index status...")

    api_keys = config.gemini_api_keys
    if not api_keys:
         console.print("[red]Error: Gemini API Key not configured.[/red]")
         return

    # Pre-calculate stats
    already_indexed = 0
    to_index = []
    
    for file_path in files:
        if db.document_exists(file_path.name):
            # Check if path needs updating
            stored_path = db.get_document_path(file_path.name)
            full_path = str(file_path.absolute())
            if stored_path != full_path:
                to_index.append(file_path) # Needs update
            else:
                already_indexed += 1
        else:
            to_index.append(file_path)

    console.print(f"Total: {len(files)} | Indexed: {already_indexed} | To Index: {len(to_index)}")
    
    if not to_index:
        console.print("[green]All summaries are already indexed.[/green]")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task_id = progress.add_task("Indexing...", total=len(to_index))

        if len(api_keys) > 1:
            # Use manual executor management
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(api_keys))
            futures = []
            try:
                for file_path in to_index:
                    # Pass ALL keys to the worker, let it manage retries
                    futures.append(executor.submit(_index_single_book, file_path, api_keys))
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        progress.advance(task_id)
                        progress.refresh() # Force refresh
                        if verbose or result["status"] == "error" or result.get("path_updated"):
                             _print_index_result(progress.console, result)
                        
                        # Check for global exhaustion
                        if result["status"] == "skipped" and "All API Keys exhausted" in result.get("message", ""):
                            progress.console.print("[bold red]CRITICAL: All API Keys have been exhausted (Daily Limits). Stopping process.[/bold red]")
                            progress.console.print(f"[red]Please check the logs for details: {config.log_file}[/red]")
                            executor.shutdown(wait=False, cancel_futures=True)
                            return
                    except Exception as e:
                        error_msg = f"Critical worker error: {e}"
                        progress.console.print(f"[red]{error_msg}[/red]")
                        logger.error(error_msg, exc_info=True)
            finally:
                executor.shutdown(wait=False)
        else:
            for file_path in to_index:
                # Check exhaustion before starting next
                if len(config.exhausted_keys) == len(api_keys):
                     progress.console.print("[bold red]CRITICAL: All API Keys have been exhausted (Daily Limits). Stopping process.[/bold red]")
                     progress.console.print(f"[red]Please check the logs for details: {config.log_file}[/red]")
                     break

                result = _index_single_book(file_path, api_keys)
                progress.advance(task_id)
                progress.refresh()
                if verbose or result["status"] == "error" or result.get("path_updated"):
                     _print_index_result(progress.console, result)
                
                if result["status"] == "skipped" and "All API Keys exhausted" in result.get("message", ""):
                     progress.console.print("[bold red]CRITICAL: All API Keys have been exhausted (Daily Limits). Stopping process.[/bold red]")
                     progress.console.print(f"[red]Please check the logs for details: {config.log_file}[/red]")
                     break

def _print_index_result(console, result, verbose: bool = False):
    if result["status"] == "skipped":
        return

    color = "green" if result["status"] == "success" else "red"
    if result.get("path_updated"):
        color = "yellow"
        
    icon = "✓"
    if result["status"] == "error":
        icon = "✗"
    elif result.get("path_updated"):
        icon = "↻"

    msg = f"[{color}]{icon} {result['file']}[/{color}]"
    if result.get("masked_key"):
        msg += f" [dim]({result['masked_key']})[/dim]"
    if result.get("chunks"):
        msg += f" | {result['chunks']} chunks"
    if result.get("message"):
        # Highlight specific errors
        error_msg = result['message']
        if "429" in error_msg or "ResourceExhausted" in error_msg:
             base_msg = "[bold red]Rate Limit Exceeded (Retried)[/bold red]"
             if "(Key:" in error_msg:
                 # Extract key part safely
                 try:
                     key_part = error_msg.split("(Key:")[-1].split(")")[0].strip()
                     error_msg = f"{base_msg} [yellow](Key: {key_part})[/yellow]"
                 except:
                     error_msg = base_msg
             else:
                 error_msg = base_msg
        msg += f" | {error_msg}"
        
    if verbose and result.get("full_error"):
        msg += f"\n[dim red]Full Error: {result['full_error']}[/dim red]"

    console.print(msg)
    
    # Log result to file
    log_msg = f"{result['status'].upper()}: {result['file']}"
    if result.get("message"):
        log_msg += f" - {result['message']}"
    
    if result["status"] == "error":
        logger.error(log_msg)
    else:
        logger.info(log_msg)

def _index_single_book(file_path: Path, api_keys: list[str]) -> dict:
    from grimoire import db
    import random
    
    # Extract PDF name from summary filename: summary_X.json -> X
    pdf_name = file_path.name.replace("summary_", "").replace(".json", "")

    # Filter out exhausted keys
    available_keys = [k for k in api_keys if k not in config.exhausted_keys]
    
    if not available_keys:
        return {"status": "skipped", "file": pdf_name, "message": "All API Keys exhausted (Daily Limit)"}
    
    # Calculate absolute path for the summary file
    full_path = str(file_path.absolute())

    if db.document_exists(file_path.name):
        # Check if path needs updating
        stored_path = db.get_document_path(file_path.name)
        if stored_path != full_path:
            db.update_document_path(file_path.name, full_path)
            return {"status": "success", "file": pdf_name, "path_updated": True, "message": "Path updated"}
        
        return {"status": "skipped", "file": pdf_name}
    
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            summary = BookSummary(**data)
    except Exception as e:
        return {"status": "error", "file": pdf_name, "message": f"Failed to load: {e}"}

    documents = []
    metadatas = []
    ids = []

    # Common Metadata (Parental)
    base_metadata = {
        "source": pdf_name,
        "filename": file_path.name,
        "full_path": str(file_path.absolute()),
        "title": summary.header.title,
        "authors": ", ".join(summary.header.authors),
        "category": summary.header.category,
        "keywords": ", ".join(summary.header.keywords)
    }

    # Chunk 1: Central Thesis
    documents.append(f"Central Thesis: {summary.central_thesis}")
    metadatas.append({**base_metadata, "chunk_type": "central_thesis"})
    ids.append(f"{pdf_name}_thesis")

    # Chunk 2: Structure/Chapters
    for i, chapter in enumerate(summary.structure_content):
        content = f"Chapter: {chapter.chapter_title}\n{chapter.summary}"
        documents.append(content)
        metadatas.append({**base_metadata, "chunk_type": "chapter", "chapter_title": chapter.chapter_title})
        ids.append(f"{pdf_name}_chapter_{i}")

    # Chunk 3: Key Concepts
    for i, concept in enumerate(summary.key_concepts):
        content = f"Concept: {concept.term}\nDefinition: {concept.definition}"
        documents.append(content)
        metadatas.append({**base_metadata, "chunk_type": "concept", "term": concept.term})
        ids.append(f"{pdf_name}_concept_{i}")

    # Chunk 4: Practical System
    if summary.practical_system:
        content = f"Practical System: {summary.practical_system.description}\n"
        if summary.practical_system.tools:
            content += f"Tools: {', '.join(summary.practical_system.tools)}\n"
        if summary.practical_system.rituals:
            content += f"Rituals: {', '.join(summary.practical_system.rituals)}"
        
        documents.append(content)
        metadatas.append({**base_metadata, "chunk_type": "practical_system"})
        ids.append(f"{pdf_name}_practical")

    # Chunk 5: Critical Analysis
    content = f"Critical Analysis:\nRelevance: {summary.critical_analysis.relevance}\nTarget Audience: {summary.critical_analysis.target_audience}"
    documents.append(content)
    metadatas.append({**base_metadata, "chunk_type": "critical_analysis"})
    ids.append(f"{pdf_name}_analysis")

    if not documents:
        return {"status": "skipped", "file": pdf_name, "message": "No documents to index"}

    # Retry Loop for Embeddings
    max_retries = 3
    retry_count = 0
    
    while available_keys:
        # Pick a key (randomly to distribute load)
        current_key = key_manager.get_best_key()
        if not current_key:
             return {"status": "skipped", "file": pdf_name, "message": "All API Keys exhausted (Daily Limit)"}
        
        try:
            # Generate embeddings in parallel (outside the lock)
            embeddings = db.generate_embeddings(documents, current_key)
            
            # Add documents with pre-calculated embeddings (inside the lock)
            db.add_documents(documents, metadatas, ids, embeddings=embeddings)
            return {"status": "success", "file": pdf_name, "chunks": len(documents), "masked_key": f"...{current_key[-4:]}"}
        
        except Exception as e:
            action, msg = handle_api_error(e, current_key, pdf_name)
            logger.error(f"Full error for {pdf_name}: {str(e)}", exc_info=True) # Added exc_info for full stack trace
            logger.warning(f"Error processing {pdf_name}: {msg}")

            if action == Action.ROTATE_KEY:
                key_manager.mark_exhausted(current_key)
                if current_key in available_keys:
                    available_keys.remove(current_key)
                logger.error(f"{msg} - Rotating key.")
                continue # Try next key

            elif action == Action.RETRY:
                retry_count += 1
                if retry_count <= max_retries:
                    logger.info(f"{msg} - Retrying ({retry_count}/{max_retries})...")
                    time.sleep(2 * retry_count)
                    continue
                else:
                    logger.error(f"Max retries exceeded for {pdf_name}.")
                    # If max retries exceeded for 429, rotate key
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                         logger.warning(f"Max retries for 429 exceeded. Assuming quota exhaustion for key ...{current_key[-4:]}. Rotating.")
                         key_manager.mark_exhausted(current_key)
                         if current_key in available_keys:
                             available_keys.remove(current_key)
                         retry_count = 0
                         continue
                    return {"status": "error", "file": pdf_name, "message": f"Max retries exceeded: {msg}", "full_error": str(e), "masked_key": f"...{current_key[-4:]}"}
            
            elif action == Action.SKIP_FILE:
                return {"status": "error", "file": pdf_name, "message": msg}
            
            else:
                return {"status": "error", "file": pdf_name, "message": msg}
    
    # If we exited the loop, it means we ran out of keys
    return {"status": "skipped", "file": pdf_name, "message": "All keys exhausted."}
def get_summary_json(pdf_name: str) -> Optional[dict]:
    """Retrieves the full JSON summary for a given PDF name."""
    summary_path = config.summaries_dir / f"summary_{pdf_name}.json"
    if not summary_path.exists():
        # Try finding it without "summary_" prefix or .json if user passed raw name
        candidates = list(config.summaries_dir.glob(f"*{pdf_name}*"))
        if not candidates:
            return None
        summary_path = candidates[0]

    try:
        with open(summary_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load summary {summary_path}: {e}")
        return None

def ask_question(query: str) -> str:
    """Answers a question using RAG."""
    from grimoire import db
    
    # 1. Search for relevant chunks (Fetch more to optimize context)
    results = db.query_documents(query, n_results=1000)
    
    if not results['documents'] or not results['documents'][0]:
        return "I couldn't find any relevant information in your library to answer that question."
    
    # 2. Prepare for Dynamic Context Construction
    all_documents = results['documents'][0]
    all_metadatas = results['metadatas'][0]
    
    # Get best key for token counting and generation
    key = key_manager.get_best_key()
    if not key:
        return "Error: No API keys available."

    try:
        client = genai.Client(api_key=key)
        tpm_limit = config.rate_limits["tpm"]
        safe_tpm_limit = int(tpm_limit * 0.8)
        
        # Binary search or iterative pruning could work. 
        # Iterative pruning from the end (least relevant) is safer for relevance.
        
        current_n = len(all_documents)
        step = 100
        
        while current_n > 0:
            # Slice documents
            current_docs = all_documents[:current_n]
            current_metas = all_metadatas[:current_n]
            
            context_parts = []
            for i, doc in enumerate(current_docs):
                meta = current_metas[i]
                source = meta.get('title', 'Unknown Source')
                context_parts.append(f"Source: {source}\nContent: {doc}")
            
            context = "\n\n".join(context_parts)
            
            prompt = f"""You are a wise and knowledgeable Grimoire, an expert in the esoteric arts.
Use the following context from the user's library to answer their question.
If the answer is not in the context, state that you cannot find it in the library, but you may offer general knowledge if explicitly asked.
Focus on being pedagogical, clear, and accurate to the provided texts.

Context:
{context}

Question: {query}

Answer:"""

            # Count tokens
            try:
                count_resp = client.models.count_tokens(
                    model=config.model_name,
                    contents=prompt
                )
                total_tokens = count_resp.total_tokens
                
                logger.info(f"Context size: {current_n} snippets. Total tokens: {total_tokens}. Limit: {safe_tpm_limit}")
                
                if total_tokens <= safe_tpm_limit:
                    # Safe to proceed
                    # Acquire rate limit (actual count + buffer for output)
                    key_manager.acquire(key, estimated_tokens=total_tokens + 500)
                    
                    response = client.models.generate_content(
                        model=config.model_name,
                        contents=prompt
                    )
                    return response.text
                
                else:
                    # Prune and retry
                    logger.info(f"Token limit exceeded ({total_tokens} > {safe_tpm_limit}). Pruning {step} snippets.")
                    current_n -= step
                    if current_n < 0: current_n = 0 # Should not happen with loop condition but safety first
            
            except Exception as e:
                logger.error(f"Error during token counting/generation: {e}")
                # If it's a rate limit error during counting, we might want to back off or try another key?
                # For now, let's assume it's a hard error and break or return
                return f"An error occurred while optimizing context: {e}"

        return "Error: Could not construct a context within token limits."

    except Exception as e:
        logger.error(f"Error initializing client or process: {e}")
        return f"An error occurred: {e}"

def repair_library(list_file_path: str, timeout: int = 60, verbose: bool = False):
    """Attempts to repair PDFs listed in the file."""
    file_path = Path(list_file_path)
    if not file_path.exists():
        console.print(f"[red]File not found: {file_path}[/red]")
        return

    # Output files
    success_file = file_path.parent / "repaired_success.txt"
    failed_file = file_path.parent / "repaired_failed.txt"
    
    console.print(f"[bold]Reading from {file_path}...[/bold]")
    pdf_paths = parse_library_list(file_path)
    console.print(f"Found {len(pdf_paths)} PDFs to repair.")

    api_keys = config.gemini_api_keys
    if not api_keys:
        console.print("[red]Error: Gemini API Key not configured.[/red]")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task_id = progress.add_task("Repairing...", total=len(pdf_paths))
        
        for pdf_path in pdf_paths:
            result = _repair_single_pdf(pdf_path, timeout, api_keys)
            progress.advance(task_id)
            
            color = "green" if result["status"] == "success" else "red"
            icon = "✓" if result["status"] == "success" else "✗"
            msg = f"[{color}]{icon} {pdf_path.name}[/{color}] - {result['message']}"
            progress.console.print(msg)
            
            if result["status"] == "success":
                with open(success_file, "a") as f:
                    f.write(f"{pdf_path}\n")
            else:
                with open(failed_file, "a") as f:
                    f.write(f"{pdf_path}\n")

def _repair_single_pdf(pdf_path: Path, timeout: int, api_keys: list[str]) -> dict:
    if not pdf_path.exists():
        return {"status": "error", "message": "File not found"}
        
    temp_path = pdf_path.with_suffix(".repaired.pdf")
    
    # 1. Run Ghostscript
    cmd = [
        "gs",
        "-o", str(temp_path),
        "-sDEVICE=pdfwrite",
        "-dPDFSETTINGS=/default",
        "-dNOPAUSE",
        "-dBATCH",
        str(pdf_path)
    ]
    
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        pass
    except Exception as e:
        return {"status": "error", "message": f"Ghostscript failed: {e}"}
        
    if not temp_path.exists() or temp_path.stat().st_size == 0:
         return {"status": "error", "message": "Ghostscript did not produce a valid file"}

    # 2. Verify with Gemini
    prompt_text = "Please provide a one-paragraph summary of this document."
    
    key = key_manager.get_best_key()
    if not key:
        if temp_path.exists(): temp_path.unlink()
        return {"status": "error", "message": "No API keys available"}
        
    try:
        client = genai.Client(api_key=key)
        with open(temp_path, "rb") as f:
            pdf_data = f.read()
            
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[
                types.Part.from_bytes(data=pdf_data, mime_type='application/pdf'),
                prompt_text
            ]
        )
        
        if response.text:
            # Success!
            bak_path = pdf_path.with_suffix(".pdf.bak")
            shutil.move(str(pdf_path), str(bak_path))
            shutil.move(str(temp_path), str(pdf_path))
            return {"status": "success", "message": "Repaired and verified"}
        else:
            if temp_path.exists(): temp_path.unlink()
            return {"status": "error", "message": "Gemini verification failed (no text)"}
            
    except Exception as e:
        if temp_path.exists(): temp_path.unlink()
        return {"status": "error", "message": f"Verification failed: {e}"}

