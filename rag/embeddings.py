from google import genai
from dotenv import load_dotenv
import os
import time

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=API_KEY)



# Call pacing only. The MODEL IS UNCHANGED (gemini-embedding-001) so that the
# chunker comparison moves exactly one variable. These settings govern how fast
# requests are sent, not what is computed, after the free-tier limit returned
# 429 RESOURCE_EXHAUSTED during bulk ingest.

MIN_INTERVAL_SECONDS = 1.1
MAX_RETRIES = 5

_last_call = [0.0]


def create_embedding(text):
    """Gemini vachi embedding pannu. Throttled + retried on 429."""

    for attempt in range(MAX_RETRIES):

        wait = MIN_INTERVAL_SECONDS - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)

        try:
            result = client.models.embed_content(
                model="gemini-embedding-001",
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
