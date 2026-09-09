"""
Thin wrapper around the Gemini REST API. Two model tiers are used on purpose:
- AGENT_MODEL: fast/cheap, used for classification, reply drafting, escalation.
- JUDGE_MODEL: a different (stronger) model, used only for LLM-as-judge scoring,
  to reduce (not eliminate -- same provider) self-preference bias. See DECISIONS.md
  and report.md ("what's misleading") for the honest limitation here.

Multiple free-tier API keys are round-robined (GEMINI_API_KEYS, comma-separated)
to work around per-key rate limits -- each key is its own free-tier quota, so
spreading calls across them raises effective throughput without changing what's
being asked of the model. Falls back to a single GEMINI_API_KEY if that's all
that's set. See DECISIONS.md.
"""
import itertools
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_keys_csv = os.environ.get("GEMINI_API_KEYS")
if _keys_csv:
    API_KEYS = [k.strip() for k in _keys_csv.split(",") if k.strip()]
else:
    _single = os.environ.get("GEMINI_API_KEY")
    API_KEYS = [_single] if _single else []

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

AGENT_MODEL = os.environ.get("AGENT_MODEL", "gemini-flash-lite-latest")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemini-flash-lite-latest")

_key_cycle = itertools.cycle(API_KEYS) if API_KEYS else None


class LLMError(RuntimeError):
    pass


def _call(model: str, prompt: str, system: str = None, json_schema: dict = None,
          temperature: float = 0.2, max_retries: int = None) -> str:
    if not API_KEYS:
        raise LLMError("no Gemini API key set (GEMINI_API_KEY or GEMINI_API_KEYS in .env)")
    if max_retries is None:
        max_retries = max(8, 4 * len(API_KEYS))

    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature},
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    if json_schema:
        body["generationConfig"]["responseMimeType"] = "application/json"
        body["generationConfig"]["responseSchema"] = json_schema

    last_err = None
    for attempt in range(max_retries):
        key = next(_key_cycle)
        url = f"{BASE_URL}/{model}:generateContent?key={key}"
        try:
            resp = requests.post(url, json=body, timeout=60)
            if resp.status_code == 429:
                # rotate to the next key, but a free-tier RPM limit doesn't
                # clear in under a second -- once we've cycled through every
                # key without success, back off for real before trying again.
                last_err = LLMError("429 rate limited")
                if (attempt + 1) % len(API_KEYS) == 0:
                    time.sleep(min(30, 3 * (2 ** (attempt // len(API_KEYS)))))
                else:
                    time.sleep(0.5)
                continue
            resp.raise_for_status()
            data = resp.json()
            cands = data.get("candidates")
            if not cands:
                raise LLMError(f"no candidates in response: {data}")
            parts = cands[0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts)
            if not text.strip():
                raise LLMError(f"empty text in response: {data}")
            return text
        except (requests.RequestException, LLMError, KeyError) as e:
            last_err = e
            time.sleep(1.5 * (attempt // len(API_KEYS) + 1))
    raise LLMError(f"LLM call failed after {max_retries} retries across {len(API_KEYS)} key(s): {last_err}")


def generate_text(prompt: str, system: str = None, model: str = None, temperature: float = 0.4) -> str:
    return _call(model or AGENT_MODEL, prompt, system=system, temperature=temperature).strip()


def generate_json(prompt: str, schema: dict, system: str = None, model: str = None,
                   temperature: float = 0.1) -> dict:
    raw = _call(model or AGENT_MODEL, prompt, system=system, json_schema=schema, temperature=temperature)
    return json.loads(raw)
