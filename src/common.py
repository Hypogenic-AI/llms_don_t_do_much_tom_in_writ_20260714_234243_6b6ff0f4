"""Shared LLM client with disk caching, retry/backoff, and concurrency.

All model calls in this project go through `chat()`, which caches responses on
disk keyed by a hash of (model, messages, sampling params). This makes the whole
pipeline reproducible and cheap to re-run: a second run is a pure cache read.
"""
import os, json, time, hashlib, threading, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

SEED = 42
random.seed(SEED)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

_client = OpenAI(base_url="https://openrouter.ai/api/v1",
                 api_key=os.environ["OPENROUTER_KEY"])
_cache_lock = threading.Lock()

# Model roster (kept in distinct families to avoid self-preference / shortcut inflation)
GEN_MODEL      = "openai/gpt-4.1-mini"            # primary generator
GEN_MODEL_ALT  = "meta-llama/llama-3.3-70b-instruct"  # replication generator
JUDGE_MODEL    = "anthropic/claude-sonnet-4.5"    # cross-family quality judge
AUDIENCE_MODEL = "google/gemini-2.5-flash"        # separate audience/listener model


def _key(model, messages, temperature, max_tokens):
    blob = json.dumps({"m": model, "msg": messages, "t": temperature,
                       "mt": max_tokens}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def chat(model, messages, temperature=0.7, max_tokens=1200, _retries=6):
    """Cached, retrying chat completion. Returns the response text (str)."""
    k = _key(model, messages, temperature, max_tokens)
    path = os.path.join(CACHE_DIR, k + ".json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["content"]
    last = None
    for i in range(_retries):
        try:
            r = _client.chat.completions.create(
                model=model, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
                extra_body={"seed": SEED})
            content = r.choices[0].message.content
            if content is None or not content.strip():
                raise ValueError("empty content")
            rec = {"content": content, "model": model, "temperature": temperature}
            with _cache_lock, open(path, "w") as f:
                json.dump(rec, f)
            return content
        except Exception as e:
            last = e
            time.sleep(min(2 ** i + random.random(), 40))
    raise RuntimeError(f"chat failed after {_retries} tries: {last}")


def pmap(fn, items, workers=12, desc=""):
    """Threaded map preserving input order; prints progress."""
    results = [None] * len(items)
    done = [0]
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, it): i for i, it in enumerate(items)}
        for fut in as_completed(futs):
            i = futs[fut]
            results[i] = fut.result()
            with lock:
                done[0] += 1
                if done[0] % 10 == 0 or done[0] == len(items):
                    print(f"  [{desc}] {done[0]}/{len(items)}", flush=True)
    return results
