"""Embeddings client — LiteLLM's OpenAI-compatible /embeddings endpoint.

Plain httpx rather than the openai SDK: the SDK always sends ``encoding_format``,
and LiteLLM rejects that parameter (like ``dimensions``) for Fireworks models.

So qwen3-embedding-8b comes back at its native 4096 dims.  The model is
Matryoshka-trained, so keeping the first ``dims`` values and L2-renormalising is
equivalent to asking for fewer dims server-side.
"""

import math

import httpx

_BATCH_SIZE = 64
_TIMEOUT_SECONDS = 60


class EmbeddingClient:
    """Turns text into unit-length vectors of a fixed width."""

    def __init__(
        self,
        *,
        model: str,
        dims: int,
        base_url: str,
        api_key: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self.dims = dims
        self._url = f"{base_url.rstrip('/')}/embeddings"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._transport = transport

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed *texts* in order.  Raises on any API error or a malformed response."""
        vectors: list[list[float]] = []
        async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS, transport=self._transport) as client:
            for start in range(0, len(texts), _BATCH_SIZE):
                batch = [t or " " for t in texts[start : start + _BATCH_SIZE]]
                response = await client.post(
                    self._url, headers=self._headers, json={"model": self.model, "input": batch}
                )
                if response.status_code != 200:
                    raise RuntimeError(
                        f"embeddings: HTTP {response.status_code} — {response.text[:300]}"
                    )
                data = sorted(response.json().get("data") or [], key=lambda d: d["index"])
                if len(data) != len(batch):
                    raise RuntimeError(
                        f"embeddings: sent {len(batch)} inputs, got {len(data)} vectors back"
                    )
                vectors.extend(_truncate(d["embedding"], self.dims) for d in data)
        return vectors

    async def embed_one(self, text: str) -> list[float]:
        (vector,) = await self.embed([text])
        return vector


def _truncate(vector: list[float], dims: int) -> list[float]:
    if len(vector) < dims:
        raise RuntimeError(f"embeddings: model returned {len(vector)} dims, need {dims}")
    head = vector[:dims]
    norm = math.sqrt(sum(x * x for x in head))
    if norm == 0:
        raise RuntimeError("embeddings: model returned a zero vector")
    return [x / norm for x in head]
