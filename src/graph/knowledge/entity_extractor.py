import json
import re

from src.llm import llm
from src.observability.logging import get_logger

logger = get_logger()

ENTITY_EXTRACTION_PROMPT = """Extract all named entities from this text. Classify each entity into one of: Person, Organization, Location, Concept, Document, Policy, Regulation, Technology, Date.

Return a JSON array of objects with: name (string), type (one of the above types).

Text:
{user_data}"""


def _parse_llm_json(raw: str) -> list[dict]:
    """Parse LLM JSON output, handling markdown code fences."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw)
    cleaned = re.sub(r"```\s*$", "", cleaned.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError, ValueError:
        return []


async def extract_entities(text: str) -> list[dict]:
    """Extract named entities from text using LLM."""
    from src.generation.prompts.safe_format import (
        ANTI_INJECTION_SYSTEM_PREAMBLE,
        format_prompt_with_user_data,
    )

    # Truncate very long texts to avoid token limits
    truncated = text[:4000] if len(text) > 4000 else text

    prompt = format_prompt_with_user_data(ENTITY_EXTRACTION_PROMPT, user_data=truncated)
    response = await llm.ainvoke(
        [
            (
                "system",
                ANTI_INJECTION_SYSTEM_PREAMBLE
                + " Extract named entities precisely as JSON.",
            ),
            ("human", prompt),
        ]
    )

    entities = _parse_llm_json(response.content)

    # Deduplicate by name (case-insensitive)
    seen = set()
    unique: list[dict] = []
    for e in entities:
        name = e.get("name", "").strip()
        if name and name.lower() not in seen:
            seen.add(name.lower())
            unique.append(e)

    logger.info("entities_extracted", count=len(unique))
    return unique
