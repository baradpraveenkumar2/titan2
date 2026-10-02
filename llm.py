"""
llm.py - thin wrapper around the Groq API (OpenAI-compatible chat completions).

The LLM never sees the raw CSV. The app sends it a compact JSON "fact sheet"
computed by engine.py, and the system prompt forces it to answer only from
those facts (no invented numbers).
"""
from __future__ import annotations

import json
import os

DEFAULT_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

SYSTEM_PROMPT = """You are a senior FMCG revenue-growth-management analyst embedded in a retailer analytics dashboard.
You are given a JSON FACT SHEET computed from the user's current filtered data.

Rules:
1. Use ONLY numbers that appear in the fact sheet. Never invent or estimate figures. If something is not in the
   fact sheet, say it is not available and suggest which tab/filter of the app would show it.
2. Be concise and specific: lead with the answer, then 2-4 supporting bullets with numbers.
3. Explain elasticity in plain language (e.g. "a 10% price cut lifts units by roughly X%").
4. Trade ROI = (incremental gross profit - trade investment) / trade investment. Negative ROI means the spend
   did not pay back through incremental gross profit.
5. The data is SYNTHETIC illustrative data - mention this only if the user asks about data quality.
6. End actionable answers with a short 'Recommended next step' line."""


def get_client(api_key: str):
    from groq import Groq   # imported lazily so the rest of the app works without the package/key
    return Groq(api_key=api_key)


def build_messages(fact_sheet: dict, history: list[dict], question: str) -> list[dict]:
    ctx = json.dumps(fact_sheet, default=str, separators=(",", ":"))
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": "FACT SHEET (JSON):\n" + ctx}]
    msgs += history[-8:]          # keep the last few turns only
    msgs.append({"role": "user", "content": question})
    return msgs


def ask(api_key: str, model: str, messages: list[dict], temperature: float = 0.2, max_tokens: int = 1100) -> str:
    if not api_key:
        return "No Groq API key found. Add it in the sidebar, in .streamlit/secrets.toml, or as the GROQ_API_KEY environment variable."
    try:
        client = get_client(api_key)
        resp = client.chat.completions.create(model=model, messages=messages,
                                              temperature=temperature, max_tokens=max_tokens)
        return resp.choices[0].message.content
    except ImportError:
        return "The `groq` package is not installed. Run: pip install groq"
    except Exception as exc:   # network / auth / model-not-found / rate-limit
        return f"Groq API error: {exc}"


EXEC_SUMMARY_PROMPT = (
    "Write an executive summary for the current selection with these sections: "
    "1) Performance snapshot, 2) Forecast outlook, 3) Price elasticity & what drives sales, "
    "4) Promotion & trade ROI - what to scale and what to cut, 5) Inventory risks, 6) Three prioritised actions."
)
