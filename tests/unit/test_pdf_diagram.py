"""Unit tests for PDF diagram extraction and dual PDF parsing."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


class _FakeSettings:
    """Minimal settings stub to avoid importing the real Settings (which triggers DB init)."""

    pdf_extract_diagrams = True
    pdf_diagram_render_dpi = 200
    pdf_diagram_render_matrix = 2.0
    pdf_diagram_min_region_ratio = 0.05
    pdf_diagram_max_regions_per_page = 5
    pdf_diagram_image_min_bytes = 2_000
    pdf_diagram_prompt = (
        "Describe this diagram in detail. Extract all visual elements, labels, "
        "and data. Use markdown formatting for clarity."
    )
    pdf_vision_page_prompt = (
        "Describe the content of this PDF page in detail. Extract text, "
        "describe diagrams, tables, and visual elements. Use markdown."
    )
    vision_max_tokens = 4096
    openai_vision_model = "gpt-4o"


def _make_mock_vision_response(content: str) -> MagicMock:
    """Create a mock LLM response mimicking LangChain AIMessage."""
    msg = MagicMock()
    msg.content = content
    msg.response_metadata = {
        "token_usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
        }
    }
    return msg


# ---------------------------------------------------------------------------
# PDFDiagramExtractor Tests
# ---------------------------------------------------------------------------

# Patch settings at the module level before importing PDFDiagramExtractor,
# since the extractor reads settings at init time.
_settings_patch = patch(
    "src.ingestion.parsers.pdf_diagram.settings",
    new=_FakeSettings(),
)
_settings_patch.start()


class PDFDiagramExtractorTests(unittest.IsolatedAsyncioTestCase):
    """Tests for PDFDiagramExtractor."""

    def setUp(self) -> None:
        self.mock_vision = MagicMock()
        self.mock_vision.ainvoke = AsyncMock()
        self._vision_patcher = patch(
            "src.ingestion.parsers.pdf_diagram.vision_llm",
            new=self.mock_vision,
        )
        self._vision_patcher.start()

    def tearDown(self) -> None:
        self._vision_patcher.stop()

    # --- init ---

    def test_loads_settings(self) -> None:
        from src.ingestion.parsers.pdf_diagram import PDFDiagramExtractor

        ext = PDFDiagramExtractor()
        self.assertEqual(ext.render_matrix, 2.0)
        self.assertEqual(ext.image_min_bytes, 2_000)
        self.assertEqual(ext.max_regions, 5)
        self.assertIn("diagram", ext.diagram_prompt.lower())
        self.assertIn("page", ext.vision_page_prompt.lower())

    # --- extract_page_as_image ---

    async def test_returns_none_for_nonexistent_file(self) -> None:
        from src.ingestion.parsers.pdf_diagram import PDFDiagramExtractor

        ext = PDFDiagramExtractor()
        result = await ext.extract_page_as_image(Path("/nonexistent/file.pdf"), 0)
        self.assertIsNone(result)

    # --- _analyze_page_image ---

    async def test_calls_vision_llm(self) -> None:
        from src.ingestion.parsers.pdf_diagram import PDFDiagramExtractor

        self.mock_vision.ainvoke.return_value = _make_mock_vision_response(
            "A diagram showing data flow"
        )
        ext = PDFDiagramExtractor()
        result = await ext._analyze_page_image(b"\x89PNGfake", page_number=1)
        self.assertEqual(result, "A diagram showing data flow")
        self.mock_vision.ainvoke.assert_called_once()
        call_args = self.mock_vision.ainvoke.call_args
        messages = call_args[0][0]
        self.assertEqual(len(messages), 1)
        content = messages[0].content
        self.assertTrue(any(part.get("type") == "image_url" for part in content))

    async def test_returns_none_on_error(self) -> None:
        from src.ingestion.parsers.pdf_diagram import PDFDiagramExtractor

        self.mock_vision.ainvoke = AsyncMock(side_effect=Exception("API error"))
        ext = PDFDiagramExtractor()
        result = await ext._analyze_page_image(b"bad", page_number=1)
        self.assertIsNone(result)

    async def test_returns_none_on_empty_content(self) -> None:
        from src.ingestion.parsers.pdf_diagram import PDFDiagramExtractor

        self.mock_vision.ainvoke.return_value = _make_mock_vision_response("")
        ext = PDFDiagramExtractor()
        result = await ext._analyze_page_image(b"\x89PNGfake", page_number=1)
        self.assertIsNone(result)

    # --- _analyze_diagram_region ---

    async def test_includes_page_number_in_prompt(self) -> None:
        from src.ingestion.parsers.pdf_diagram import PDFDiagramExtractor

        self.mock_vision.ainvoke.return_value = _make_mock_vision_response(
            "Flowchart with 3 nodes"
        )
        ext = PDFDiagramExtractor()
        result = await ext._analyze_diagram_region(
            b"fakeimage", region_label="diagram 1", page_number=5
        )
        self.assertEqual(result, "Flowchart with 3 nodes")
        call_args = self.mock_vision.ainvoke.call_args
        messages = call_args[0][0]
        text_parts = [p for p in messages[0].content if p.get("type") == "text"]
        self.assertTrue(any("page 5" in p["text"] for p in text_parts))

    # --- extract_page_diagrams ---

    async def test_returns_empty_for_nonexistent_file(self) -> None:
        from src.ingestion.parsers.pdf_diagram import PDFDiagramExtractor

        ext = PDFDiagramExtractor()
        result = await ext.extract_page_diagrams(Path("/nonexistent/file.pdf"), 0)
        self.assertEqual(result, [])

    # --- extract_all_pages ---

    async def test_returns_empty_list_for_invalid_file(self) -> None:
        from src.ingestion.parsers.pdf_diagram import PDFDiagramExtractor

        ext = PDFDiagramExtractor()
        result = await ext.extract_all_pages(Path("/nonexistent/file.pdf"))
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# DocumentParser Merge Tests
# ---------------------------------------------------------------------------


class DocumentParserMergeTests(unittest.TestCase):
    """Tests for DocumentParser._merge_signals."""

    def _make_signal(self, name: str, content: str, confidence: float) -> "ParserSignal":
        from src.ingestion.parsers.base import ParserSignal

        return ParserSignal(
            name=name,
            content=content,
            confidence=confidence,
            source="test",
            metadata={"fallback_text_length": 0},
        )

    def test_merge_both_available(self) -> None:
        from src.ingestion.parsers.document import DocumentParser

        parser = DocumentParser()
        text = self._make_signal("pdf_page_llm", "Text from page 1", 0.93)
        vision = self._make_signal("pdf_vision_page_analysis", "Visual diagram", 0.95)
        result = parser._merge_signals(text, vision, page_count=1)
        self.assertIsNotNone(result)
        self.assertIn("Text from page 1", result.content)
        self.assertIn("Visual diagram", result.content)
        self.assertIn("Visual Analysis", result.content)
        self.assertEqual(result.name, "pdf_dual_text_vision")
        self.assertEqual(result.confidence, 0.95)

    def test_merge_text_only(self) -> None:
        from src.ingestion.parsers.document import DocumentParser

        parser = DocumentParser()
        text = self._make_signal("pdf_page_llm", "Text content", 0.93)
        result = parser._merge_signals(text, None, page_count=1)
        self.assertIsNotNone(result)
        self.assertEqual(result.content, "Text content")

    def test_merge_vision_only(self) -> None:
        from src.ingestion.parsers.document import DocumentParser

        parser = DocumentParser()
        vision = self._make_signal("pdf_vision_page_analysis", "Visual content", 0.95)
        result = parser._merge_signals(None, vision, page_count=1)
        self.assertIsNotNone(result)
        self.assertEqual(result.content, "Visual content")

    def test_merge_both_none(self) -> None:
        from src.ingestion.parsers.document import DocumentParser

        parser = DocumentParser()
        result = parser._merge_signals(None, None, page_count=1)
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# DocumentParser.can_parse Tests
# ---------------------------------------------------------------------------


class DocumentParserCanParseTests(unittest.TestCase):
    def test_pdf(self) -> None:
        from src.ingestion.parsers.document import DocumentParser

        self.assertTrue(DocumentParser().can_parse(Path("test.pdf")))
        self.assertTrue(DocumentParser().can_parse(Path("TEST.PDF")))

    def test_docx(self) -> None:
        from src.ingestion.parsers.document import DocumentParser

        self.assertTrue(DocumentParser().can_parse(Path("report.docx")))

    def test_unknown(self) -> None:
        from src.ingestion.parsers.document import DocumentParser

        self.assertFalse(DocumentParser().can_parse(Path("test.xyz")))
        self.assertFalse(DocumentParser().can_parse(Path("image.png")))
