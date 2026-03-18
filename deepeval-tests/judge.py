"""
Configurable judge model adapter for DeepEval.

Defaults target SiliconFlow's OpenAI-compatible chat completions API, while
remaining generic enough for other OpenAI-compatible providers.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from deepeval.models.base_model import DeepEvalBaseLLM
from openai import AsyncOpenAI, OpenAI


DEFAULT_SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_SILICONFLOW_MODEL = "Qwen/Qwen2.5-72B-Instruct"


class OpenAICompatibleJudge(DeepEvalBaseLLM):
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.0,
    ) -> None:
        if not api_key:
            raise RuntimeError(
                "Missing judge API key. Set JUDGE_API_KEY or SILICONFLOW_API_KEY."
            )
        self.model = model
        self.temperature = temperature
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    def load_model(self) -> "OpenAICompatibleJudge":
        return self

    def generate(self, prompt: str, schema: Any | None = None) -> str:
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        content = completion.choices[0].message.content
        if not content:
            raise RuntimeError("Judge model returned an empty response.")
        return content

    async def a_generate(self, prompt: str, schema: Any | None = None) -> str:
        completion = await self._async_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        content = completion.choices[0].message.content
        if not content:
            raise RuntimeError("Judge model returned an empty response.")
        return content

    def get_model_name(self) -> str:
        return self.model


@lru_cache(maxsize=1)
def get_judge_model() -> OpenAICompatibleJudge:
    api_key = os.getenv("JUDGE_API_KEY") or os.getenv("SILICONFLOW_API_KEY", "")
    base_url = os.getenv("JUDGE_BASE_URL", DEFAULT_SILICONFLOW_BASE_URL)
    model = os.getenv("JUDGE_MODEL", DEFAULT_SILICONFLOW_MODEL)
    temperature = float(os.getenv("JUDGE_TEMPERATURE", "0"))
    return OpenAICompatibleJudge(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )
