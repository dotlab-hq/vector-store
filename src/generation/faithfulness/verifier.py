import json
from pathlib import Path

from src.llm import llm
from src.generation.prompts.safe_format import (
    fence_user_data,
    ANTI_INJECTION_SYSTEM_PREAMBLE,
)
from src.observability.logging import get_logger

logger = get_logger()

CLAIM_EXTRACTION_PROMPT = """Extract all factual claims from this answer. Each claim should be a single verifiable statement.
Return a JSON array of strings, each being one claim.

Answer: {answer}"""

_PROMPT_DIR = Path(__file__).parent.parent / "prompts"
VERIFICATION_PROMPT = (_PROMPT_DIR / "faithfulness_prompt.txt").read_text()


class FaithfulnessVerifier:
    async def verify(self, response: str, context: str) -> tuple[float, int, int]:
        if not context or not response:
            return 1.0, 0, 0

        claims = await self._extract_claims(response)
        if not claims:
            return 1.0, 0, 0

        supported = await self._verify_claims(claims, context)
        total = len(claims)
        supported_count = sum(1 for s in supported if s)
        score = supported_count / total if total > 0 else 1.0

        logger.info(
            "faithfulness_verified",
            total_claims=total,
            supported_claims=supported_count,
            score=score,
        )
        return score, total, supported_count

    async def _extract_claims(self, answer: str) -> list[str]:
        fenced_answer = fence_user_data(answer)
        prompt = CLAIM_EXTRACTION_PROMPT.format(answer=fenced_answer)
        response = await llm.ainvoke(
            [
                (
                    "system",
                    f"{ANTI_INJECTION_SYSTEM_PREAMBLE} Extract claims precisely as JSON array.",
                ),
                ("human", prompt),
            ]
        )
        try:
            return json.loads(response.content)
        except json.JSONDecodeError, ValueError:
            return []

    async def _verify_claims(self, claims: list[str], context: str) -> list[bool]:
        fenced_context = fence_user_data(context)
        fenced_claims = fence_user_data(json.dumps(claims))
        prompt = VERIFICATION_PROMPT.format(
            context=fenced_context,
            claims=fenced_claims,
        )
        response = await llm.ainvoke(
            [
                (
                    "system",
                    f"{ANTI_INJECTION_SYSTEM_PREAMBLE} Verify claims against context.",
                ),
                ("human", prompt),
            ]
        )
        try:
            verifications = json.loads(response.content)
            return [v.get("supported", False) for v in verifications]
        except json.JSONDecodeError, ValueError:
            return [False] * len(claims)
