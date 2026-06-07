from __future__ import annotations

import unittest
from dataclasses import asdict

from src.ingestion.media.processor import MediaProcessingService, MediaSignal


class MediaProcessorTests(unittest.TestCase):
    def test_media_signal_score_serializes_cleanly(self) -> None:
        service = MediaProcessingService()
        signal = MediaSignal(
            name="ocr",
            content="hello world from a scanned document",
            confidence=0.87,
            source="vision",
            metadata={"page": 1},
        )

        score = service._score_signal(signal, "hello world")
        payload = asdict(score)

        self.assertEqual(payload["signal_name"], "ocr")
        self.assertGreaterEqual(payload["weighted_score"], 0)


if __name__ == "__main__":
    unittest.main()
