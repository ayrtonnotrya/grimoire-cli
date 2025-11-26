import os
from pathlib import Path
from rich.console import Console
from google import genai
from google.genai import types
from grimoire.config import config
from grimoire.schemas import BookSummary, BOOK_SUMMARY_SCHEMA
import json

console = Console()

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
    summary_path = config.summaries_dir / f"summary_{pdf_name}.md"
    return summary_path.exists()

def process_library(list_file_path: str):
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

    for pdf_path in pdf_paths:
        pdf_name = pdf_path.name
        if check_summary_exists(pdf_name):
            console.print(f"[yellow]Skipping existing summary for: {pdf_name}[/yellow]")
            continue
        
        console.print(f"[green]Processing: {pdf_name}[/green]")
        generate_summary(pdf_path)

def generate_summary(pdf_path: Path):
    """Generates a summary for the given PDF using Gemini."""
    api_key = config.gemini_api_key
    if not api_key:
        console.print("[red]Error: Gemini API Key not configured. Run 'grimoire init'.[/red]")
        return

    client = genai.Client(api_key=api_key)
    
    console.print(f"Reading {pdf_path.name}...")
    try:
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()
    except Exception as e:
        console.print(f"[red]Failed to read file: {e}[/red]")
        return

    console.print("Generating summary...")

    # Load prompt
    # Try to find it in the package templates directory
    current_dir = Path(__file__).parent
    prompt_path = current_dir / "templates" / "book_summary_prompt.md"
    
    if not prompt_path.exists():
         console.print(f"[red]Prompt file not found at {prompt_path}[/red]")
         return
    
    with open(prompt_path, "r") as f:
        prompt_text = f.read()

    try:
        response = client.models.generate_content(
            model=config.model_name,
            contents=[
                types.Part.from_bytes(
                    data=pdf_data,
                    mime_type='application/pdf',
                ),
                prompt_text
            ],
            config={
                'response_mime_type': 'application/json',
                'response_schema': BOOK_SUMMARY_SCHEMA,
            }
        )
        
        # Parse JSON
        try:
            # response.parsed might be a dict now, or None if we used a dict schema?
            # Let's rely on response.text and parse it ourselves to be safe and robust.
            summary_text = response.text
            summary_dict = json.loads(summary_text)
            summary_data = BookSummary(**summary_dict)
            
            # Save JSON
            json_path = config.summaries_dir / f"summary_{pdf_path.name}.json"
            config.summaries_dir.mkdir(parents=True, exist_ok=True)
            
            with open(json_path, "w") as f:
                f.write(summary_data.model_dump_json(indent=2))
                
            console.print(f"[bold green]JSON Summary saved to {json_path}[/bold green]")

            # Save Markdown
            md_path = config.summaries_dir / f"summary_{pdf_path.name}.md"
            save_summary_as_markdown(summary_data, md_path)
            console.print(f"[bold green]Markdown Summary saved to {md_path}[/bold green]")

        except Exception as e:
             console.print(f"[red]Failed to parse/save response: {e}[/red]")
             console.print(f"Raw text: {response.text[:500]}...")

    except Exception as e:
        console.print(f"[red]Generation failed: {e}[/red]")

def save_summary_as_markdown(summary: BookSummary, output_path: Path):
    """Converts BookSummary to Markdown and saves it."""
    md_content = f"# {summary.header.title}\n\n"
    md_content += f"**Author(s):** {', '.join(summary.header.authors)}\n"
    md_content += f"**Category:** {summary.header.category}\n"
    md_content += f"**Keywords:** {', '.join(summary.header.keywords)}\n\n"
    
    md_content += "## Central Thesis\n"
    md_content += f"{summary.central_thesis}\n\n"
    
    md_content += "## Structure and Content\n"
    for chapter in summary.structure_content:
        md_content += f"### {chapter.chapter_title}\n"
        md_content += f"{chapter.summary}\n\n"
        
    md_content += "## Key Concepts\n"
    for concept in summary.key_concepts:
        md_content += f"- **{concept.term}:** {concept.definition}\n"
    md_content += "\n"
    
    if summary.practical_system:
        md_content += "## Practical System\n"
        md_content += f"{summary.practical_system.description}\n\n"
        if summary.practical_system.tools:
            md_content += "**Tools:**\n"
            for tool in summary.practical_system.tools:
                md_content += f"- {tool}\n"
            md_content += "\n"
        if summary.practical_system.rituals:
            md_content += "**Rituals:**\n"
            for ritual in summary.practical_system.rituals:
                md_content += f"- {ritual}\n"
            md_content += "\n"
            
    md_content += "## Relevant Quotes\n"
    for quote in summary.relevant_quotes:
        page_str = f" (p. {quote.page})" if quote.page else ""
        md_content += f"> \"{quote.text}\"{page_str}\n\n"
        
    md_content += "## Critical Analysis\n"
    md_content += f"**Relevance:** {summary.critical_analysis.relevance}\n\n"
    md_content += f"**Target Audience:** {summary.critical_analysis.target_audience}\n"
    
    with open(output_path, "w") as f:
        f.write(md_content)

def index_summaries():
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

    console.print(f"Found {len(files)} summaries. Indexing with parental chunking...")

    for file_path in files:
        # Extract PDF name from summary filename: summary_X.json -> X
        pdf_name = file_path.name.replace("summary_", "").replace(".json", "")
        
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                summary = BookSummary(**data)
        except Exception as e:
            console.print(f"[red]Failed to load {file_path}: {e}[/red]")
            continue

        # Common Metadata (Parental)
        base_metadata = {
            "source": pdf_name,
            "filename": file_path.name,
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

    try:
        if documents:
            db.add_documents(documents, metadatas, ids)
            console.print(f"[bold green]Successfully indexed {len(documents)} chunks from {len(files)} books.[/bold green]")
        else:
            console.print("[yellow]No documents to index.[/yellow]")
    except Exception as e:
        console.print(f"[red]Indexing failed: {e}[/red]")



