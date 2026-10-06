from google import genai
from dotenv import load_dotenv
import os
import time

from rag import cache
from rag.observability import span

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

_client = None


def _get_client():
    """Create the Gemini client on first use, not at import time (see
    rag/generator.py's _get_client for why)."""

    global _client
    if _client is None:
        if not API_KEY:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        _client = genai.Client(api_key=API_KEY)
    return _client

# Call pacing only. The MODEL IS UNCHANGED (gemini-embedding-001) so that the
# chunker comparison moves exactly one variable. These settings govern how fast
# requests are sent, not what is computed, after the free-tier limit returned
# 429 RESOURCE_EXHAUSTED during bulk ingest.

MIN_INTERVAL_SECONDS = 1.1
MAX_RETRIES = 5

_last_call = [0.0]


EMBEDDING_MODEL = "gemini-embedding-001"


def create_embedding(text):
    """Embed `text` with Gemini. Cached (rag/cache.py), traced, throttled and
    retried on 429."""

    with span("embedding", chars=len(text)) as node:
        cached = cache.get_embedding(EMBEDDING_MODEL, text)
        if cached is not None:
            if node:
                node.set(cache_hit=True)
            return cached

        vector = _embed_uncached(text)
        cache.put_embedding(EMBEDDING_MODEL, text, vector)
        if node:
            node.set(cache_hit=False)
        return vector


def _embed_uncached(text):

    for attempt in range(MAX_RETRIES):

        wait = MIN_INTERVAL_SECONDS - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)

        try:
            result = _get_client().models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text
            )
            _last_call[0] = time.time()
            return result.embeddings[0].values

        except Exception as error:

            _last_call[0] = time.time()

            if "RESOURCE_EXHAUSTED" not in str(error) and "429" not in str(error):
                raise

            if attempt == MAX_RETRIES - 1:
                raise

            backoff = 20 * (attempt + 1)
            print(f"  [rate limited, backing off {backoff}s]", flush=True)
            time.sleep(backoff)
