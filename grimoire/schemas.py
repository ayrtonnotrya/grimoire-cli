from pydantic import BaseModel, Field
from typing import List, Optional

class BookHeader(BaseModel):
    title: str = Field(description="The full title of the book")
    authors: List[str] = Field(description="List of authors")
    category: str = Field(description="Primary category (e.g., Chaos Magic, Hermetics, Wicca)")
    keywords: List[str] = Field(description="5-10 relevant keywords")

class SearchPlan(BaseModel):
    search_queries: List[str] = Field(description="List of precise search queries to retrieve occult knowledge.")

class RitualTool(BaseModel):
    name: str = Field(description="Name of the tool")
    usage: str = Field(description="How the tool is used in the ritual")
    substitute: Optional[str] = Field(None, description="Suggested substitute if the user lacks the tool")

class RitualStep(BaseModel):
    name: str = Field(description="Name of the step")
    action: str = Field(description="Physical action to perform")
    incantation: Optional[str] = Field(None, description="Verbal incantation")
    visualization: Optional[str] = Field(None, description="Mental visualization")

class ConstructedRitual(BaseModel):
    title: str = Field(description="Title of the ritual")
    intent: str = Field(description="The specific intent of the ritual")
    timing: str = Field(description="Best timing (moon phase, day, hour)")
    tools: List[RitualTool] = Field(description="List of required tools")
    steps: List[RitualStep] = Field(description="Sequential steps of the ritual")
    closing: str = Field(description="How to close the ritual")
    expected_result: str = Field(description="What to expect as a result")

class SigilPrompt(BaseModel):
    visual_prompt: str = Field(
        description="A dense, descriptive visual prompt for the image generator. MUST be under 480 tokens. Focus on visual elements, style, and symbolism."
    )

class ChapterSummary(BaseModel):
    chapter_title: str = Field(..., description="Title of the chapter or section")
    summary: str = Field(..., description="Concise summary of the chapter's content")

class ConceptDefinition(BaseModel):
    term: str = Field(..., description="The esoteric or philosophical term")
    definition: str = Field(..., description="Definition according to the book's context")

class PracticalSystem(BaseModel):
    description: str = Field(..., description="General description of the proposed system")
    tools: List[str] = Field(default_factory=list, description="List of tools mentioned")
    rituals: List[str] = Field(default_factory=list, description="List of rituals or exercises")

class Quote(BaseModel):
    text: str = Field(..., description="The literal quote")
    page: Optional[str] = Field(None, description="Page number if available")

class CriticalAnalysis(BaseModel):
    relevance: str = Field(..., description="Relevance within the library tree/tradition")
    target_audience: str = Field(..., description="Intended audience (beginner, adept, academic)")

class BookSummary(BaseModel):
    header: BookHeader
    central_thesis: str = Field(..., description="Summary of the main purpose of the book")
    structure_content: List[ChapterSummary] = Field(..., description="Summary by chapters or sections")
    key_concepts: List[ConceptDefinition] = Field(..., description="Fundamental terms and definitions")
    practical_system: Optional[PracticalSystem] = Field(None, description="Practical system and rituals if applicable")
    relevant_quotes: List[Quote] = Field(..., description="3 to 5 relevant quotes")
    critical_analysis: CriticalAnalysis

# Explicit Schema for Gemini API (No $ref, No Optional/Null issues)
# Explicit Schema for Gemini API (No $ref, No Optional/Null issues)
BOOK_SUMMARY_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "header": {
            "type": "OBJECT",
            "properties": {
                "title": {"type": "STRING"},
                "authors": {"type": "ARRAY", "items": {"type": "STRING"}},
                "category": {"type": "STRING"},
                "keywords": {"type": "ARRAY", "items": {"type": "STRING"}}
            },
            "required": ["title", "authors", "category", "keywords"]
        },
        "central_thesis": {"type": "STRING"},
        "structure_content": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "chapter_title": {"type": "STRING"},
                    "summary": {"type": "STRING"}
                },
                "required": ["chapter_title", "summary"]
            }
        },
        "key_concepts": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "term": {"type": "STRING"},
                    "definition": {"type": "STRING"}
                },
                "required": ["term", "definition"]
            }
        },
        "practical_system": {
            "type": "OBJECT",
            "properties": {
                "description": {"type": "STRING"},
                "tools": {"type": "ARRAY", "items": {"type": "STRING"}},
                "rituals": {"type": "ARRAY", "items": {"type": "STRING"}}
            },
            "required": ["description"]
        },
        "relevant_quotes": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "text": {"type": "STRING"},
                    "page": {"type": "STRING"}
                },
                "required": ["text"]
            }
        },
        "critical_analysis": {
            "type": "OBJECT",
            "properties": {
                "relevance": {"type": "STRING"},
                "target_audience": {"type": "STRING"}
            },
            "required": ["relevance", "target_audience"]
        }
    },
    "required": ["header", "central_thesis", "structure_content", "key_concepts", "relevant_quotes", "critical_analysis"]
}

CONSTRUCTED_RITUAL_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "title": {"type": "STRING"},
        "intent": {"type": "STRING"},
        "timing": {"type": "STRING"},
        "tools": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "usage": {"type": "STRING"},
                    "substitute": {"type": "STRING"}
                },
                "required": ["name", "usage"]
            }
        },
        "steps": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "action": {"type": "STRING"},
                    "incantation": {"type": "STRING"},
                    "visualization": {"type": "STRING"}
                },
                "required": ["name", "action"]
            }
        },
        "closing": {"type": "STRING"},
        "expected_result": {"type": "STRING"}
    },
    "required": ["title", "intent", "timing", "tools", "steps", "closing", "expected_result"]
}
