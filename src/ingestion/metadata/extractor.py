import re
from pathlib import Path

from src.ingestion.loaders.base import RawDocument
from src.shared.types import Document


class MetadataExtractor:
    def extract(self, raw: RawDocument, file_path: Path) -> Document:
        title = raw.metadata.get("title", file_path.stem)
        author = raw.metadata.get("author", "")
        source_type = file_path.suffix.lower().lstrip(".")

        # Extract entities: capitalized multi-word phrases and acronyms
        entities = self._extract_entities(raw.content)

        return Document(
            id=file_path.stem,
            title=title,
            source_path=raw.source_path or str(file_path),
            source_type=source_type,
            author=author,
            tags=entities[:20],  # limit entities as tags
            metadata=raw.metadata,
        )

    def _extract_entities(self, text: str) -> list[str]:
        # Capitalized words/phrases (simple NER heuristic)
        cap_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
        acronyms = r"\b([A-Z]{2,})\b"
        entities = set()
        entities.update(re.findall(cap_pattern, text))
        entities.update(re.findall(acronyms, text))
        return list(entities)
