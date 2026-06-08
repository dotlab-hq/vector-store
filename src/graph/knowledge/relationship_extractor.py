import json
import re

from src.llm import llm
from src.observability.logging import get_logger

logger = get_logger()

RELATIONSHIP_EXTRACTION_PROMPT = """Extract all relationships between the given entities in this text.
Each relationship should connect two entities with a verb phrase.

Return a JSON array of objects with:
- source (string): the subject entity name
- target (string): the object entity name
- relationship (string): the verb phrase (e.g., "regulates", "owns", "implements")

Only extract relationships that are explicitly stated or strongly implied in the text.

Entities: {entities}

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


async def extract_relationships(text: str, entities: list[dict]) -> list[dict]:
    """Extract relationships between entities from text using LLM."""
    from src.generation.prompts.safe_format import (
        ANTI_INJECTION_SYSTEM_PREAMBLE,
        format_prompt_with_user_data,
    )

    if len(entities) < 2:
        return []

    truncated = text[:4000] if len(text) > 4000 else text
    entity_names = [e.get("name", "") for e in entities if e.get("name")]

    prompt = format_prompt_with_user_data(
        RELATIONSHIP_EXTRACTION_PROMPT,
        user_data=truncated,
        entities=", ".join(entity_names),
    )
    response = await llm.ainvoke(
        [
            (
                "system",
                ANTI_INJECTION_SYSTEM_PREAMBLE
                + " Extract relationships between entities precisely as JSON.",
            ),
            ("human", prompt),
        ]
    )

    relationships = _parse_llm_json(response.content)

    # Validate relationships reference known entities
    entity_set = {name.lower() for name in entity_names}
    valid: list[dict] = []
    for rel in relationships:
        src = rel.get("source", "").strip()
        tgt = rel.get("target", "").strip()
        rel_type = rel.get("relationship", "").strip()
        if (
            src
            and tgt
            and rel_type
            and src.lower() in entity_set
            and tgt.lower() in entity_set
        ):
            valid.append(rel)

    logger.info("relationships_extracted", count=len(valid))
    return valid
