"""
Interactive tutor GUI (full-featured)
- Uses Gemini (Google Generative) model for reasoning/generation
- Runs semantic_search_final.py for each keyword produced by the reasoner
- Appends results to reasoningmodel.txt until quiz threshold is passed
- Robust JSON extraction and debug logging
- Tkinter GUI with progress bars, dashboard, question box, history

Change summary (key points):
- CHUNKING REMOVED: final lesson synthesis uses a single API call with the entire reasoningmodel.txt (subject to MAX_PROMPT_CHARS safety guard).
- Lots of console prints + debug_log_raw checkpoints added for easier debugging.
- Subprocess UTF-8 handling already present to avoid UnicodeDecodeError.
"""
import os
import json
import ast
import re
import sys
import time
import traceback
import subprocess
import threading
import queue
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

# ------------------------- Configuration ------------------------- 

# NOTE: keep your API key in environment; do NOT hardcode in source.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Please set GEMINI_API_KEY environment variable (do NOT hardcode).")

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
REASONING_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")   # change if you want another model
GEMINI_URL = f"{GEMINI_BASE}/{REASONING_MODEL}:generateContent"

# safety: maximum characters we'll send in a single prompt (configurable via env)
MAX_PROMPT_CHARS = int(os.getenv("MAX_PROMPT_CHARS", "20000"))

BASE_DIR = Path(r"C:\Users\khush\OneDrive\Desktop\AI project")
FILE_PATH = BASE_DIR / "Temp_semantic.txt"
QUERY_PATH = BASE_DIR / "query.txt"
REASONING_MODEL_TXT = BASE_DIR / "reasoningmodel.txt"
SEMANTIC_SCRIPT = BASE_DIR / "semantic_search_final.py"  # script to run each keyword
PROGRESS_FILE = BASE_DIR / "learning_progress_tk.json"
# Directory for previous-year question papers (put PYQ files here: .txt/.json/.md/.pdf (see note))
PYQ_DIR = BASE_DIR / "pyq_papers"
PYQ_DIR.mkdir(parents=True, exist_ok=True)
# Allowed extensions to load (you may extend)
PYQ_EXTENSIONS = [".txt", ".json", ".md"]

DEBUG_RAW_FILE = BASE_DIR / "debug_model_raw.txt"

# UI thresholds
PASS_THRESHOLD = 0.7     # 70% to pass

# Reasoner call settings
REQUEST_TIMEOUT = 120
STREAM_CHUNK_TIMEOUT = 1.0

# Ensure base dir exists
BASE_DIR.mkdir(parents=True, exist_ok=True)
# Ensure reasoningmodel file exists so later appends won't fail
if not REASONING_MODEL_TXT.exists():
    REASONING_MODEL_TXT.write_text("", encoding="utf-8")

# =================================================================

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# HTTP session (do not set Authorization Bearer for Gemini; we will use x-goog-api-key per-request)
session = requests.Session()
session.headers.update({"Content-Type": "application/json"})

# ------------------------- Debug log helpers -------------------------
def debug_log_raw(raw: Any, tag: Optional[str] = None) -> None:
    """Append raw model output (any type) with timestamp to debug file (UTF-8)."""
    try:
        with open(DEBUG_RAW_FILE, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            if tag:
                f.write(f"Tag: {tag}\n")
            if isinstance(raw, (list, tuple)):
                f.write("\n".join(map(str, raw)))
            else:
                try:
                    f.write(str(raw))
                except Exception:
                    f.write(repr(raw))
            f.write("\n" + "=" * 80 + "\n\n")
    except Exception:
        logging.exception("Failed to write debug raw log")

def safe_write_text_file(path: Path, obj: Any, as_json: bool = True) -> None:
    """
    Write `obj` to `path` as plain text. If as_json=True, try json.dumps(...) with default=str
    (so non-serializable objects are stringified), otherwise just write str(obj).
    This avoids exceptions from json.dump on weird objects.
    """
    text = None
    try:
        if as_json:
            # default=str will stringify datetimes / custom objects rather than failing
            text = json.dumps(obj, indent=2, ensure_ascii=False, default=str)
        else:
            text = str(obj)
    except Exception:
        # last resort: use repr to guarantee we can write something
        try:
            text = repr(obj)
        except Exception:
            text = "<unserializable object>"
    # ensure parent dir exists
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

# ------------------------- JSON extraction -------------------------
import re, json, ast
from typing import Any

def collapse_spaced_letter_sequences(s: str) -> str:
    """
    Find sequences like "G o o d e v e n i n g" or "e x p l a n a t i o n"
    and collapse them to "Goodevening" / "explanation". Leave normal
    multi-word quoted text alone.
    """
    # This pattern matches runs of single letters separated by spaces, of length >=3:
    # e.g. "G o o d", "e x p l a n a t i o n"
    spaced_letters_pat = re.compile(r'(?:[A-Za-z](?:\s+|$)){3,}')
    
    def _collapse_match(m):
        txt = m.group(0)
        # remove whitespace between letters
        collapsed = re.sub(r'\s+', '', txt)
        return collapsed

    # Replace each spaced-letter run inside s
    return spaced_letters_pat.sub(_collapse_match, s)

def extract_json_from_text(raw: str) -> Any:
    if not raw or not isinstance(raw, str):
        raise RuntimeError("Empty or invalid model output")

    cleaned = raw.strip()
    cleaned = re.sub(r"(?is)^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"(?is)\s*```\s*$", "", cleaned)
    cleaned = cleaned.replace("`", " ")
    cleaned = re.sub(r"[ \t\f\r]+", " ", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)

    # Instead of collapsing ALL spaces inside every quoted string, iterate quotes
    # and only collapse letter-spaced sequences within them.
    def _fix_quotes(match):
        inner = match.group(1)  # contents inside quotes (no surrounding quotes)
        fixed_inner = collapse_spaced_letter_sequences(inner)
        return '"' + fixed_inner + '"'

    # process all quoted strings but safely (limit individual quoted length to avoid catastrophes)
    cleaned = re.sub(r'"([^"]{0,1200})"', _fix_quotes, cleaned)  # limit match length for safety

    # proceed with candidate extraction and parsing as before...
    def find_candidates(text: str):
        candidates = []
        pairs = [('{', '}'), ('[', ']')]
        for o, c in pairs:
            starts = [m.start() for m in re.finditer(re.escape(o), text)]
            for s in starts:
                depth = 0
                for i in range(s, len(text)):
                    if text[i] == o:
                        depth += 1
                    elif text[i] == c:
                        depth -= 1
                        if depth == 0:
                            candidates.append(text[s:i+1])
                            break
        return candidates

    candidates = find_candidates(cleaned)
    if not candidates:
        candidates = [cleaned]

    last_exc = None
    for cand in candidates:
        # quick fix: quote numeric+unit patterns
        cand_fixed = re.sub(r'([:\[,]\s*)(-?\d+(\.\d+)?)(\s*[A-Za-z/²%°]+)', r'\1"\2\4"', cand)
        try:
            return json.loads(cand_fixed)
        except Exception as e_json:
            last_exc = e_json
            try:
                return ast.literal_eval(cand_fixed)
            except Exception as e_ast:
                last_exc = e_ast
                continue

    snippet = (raw[:1000].replace("\n", "\\n"))
    raise RuntimeError(f"Could not parse JSON from model output. Last error: {last_exc}\nRaw snippet: {snippet}")

def call_reasoner(prompt: str,
                  model: Optional[str] = None,
                  max_tokens: int = 600,
                  temperature: float = 0.2,
                  endpoint: Optional[str] = None,
                  stream: bool = False,
                  try_extract_json: bool = True) -> Tuple[str, Optional[Any]]:
    """
    Unified reasoner caller for Gemini:
      - Calls Gemini generateContent endpoint using x-goog-api-key header.
      - Non-streaming (Gemini generateContent is typically called non-streaming here).
    Returns (raw_output, parsed_json_or_None).
    """
    model_to_call = model or REASONING_MODEL or "gemini-2.0-flash"
    url = endpoint or f"{GEMINI_BASE}/{model_to_call}:generateContent"

    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}

    # checkpoint
    print(f"[call_reasoner] Calling Gemini model={model_to_call}, prompt_len={len(prompt)}")
    debug_log_raw({"stage": "call_reasoner_start", "model": model_to_call, "prompt_len": len(prompt)}, tag="call_reasoner_start")

    try:
        resp = session.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    except Exception as e:
        debug_log_raw(f"HTTP exception calling Gemini: {e}", tag="gemini_http_exception")
        print("[call_reasoner] HTTP exception:", e)
        raise

    if not resp.ok:
        try:
            debug_log_raw({"status": resp.status_code, "body": resp.text}, tag="gemini_http_error")
            print("[call_reasoner] GEMINI ERROR:", resp.status_code, resp.text[:1000])
        except Exception:
            pass
        resp.raise_for_status()

    raw_output = ""
    parsed = None

    try:
        body = resp.json()
    except Exception:
        raw_output = resp.text
        debug_log_raw(f"Non-JSON response from Gemini: {raw_output[:2000]}", tag="gemini_nonjson")
        if try_extract_json:
            try:
                parsed = extract_json_from_text(raw_output)
            except Exception:
                parsed = None
        print("[call_reasoner] Non-JSON response; returning raw text")
        return raw_output, parsed

    pieces = []
    try:
        for c in body.get("candidates", []):
            content = c.get("content", {}) or {}
            parts = []
            if isinstance(content, dict):
                parts = content.get("parts") or []
            elif isinstance(content, list):
                parts = content
            for p in parts:
                if isinstance(p, dict) and "text" in p:
                    pieces.append(p["text"])
                elif isinstance(p, str):
                    pieces.append(p)
    except Exception:
        debug_log_raw({"body_parse_exc": traceback.format_exc(), "body_sample": str(body)[:2000]}, tag="gemini_parse_exc")

    raw_output = "".join(pieces).strip() if pieces else json.dumps(body)
    debug_log_raw(raw_output, tag="raw_model_output")
    print(f"[call_reasoner] Received output length {len(raw_output)}")

    if try_extract_json:
        try:
            parsed = extract_json_from_text(raw_output)
            print("[call_reasoner] Parsed JSON successfully")
            debug_log_raw({"stage": "call_reasoner_parsed", "parsed_type": type(parsed).__name__}, tag="call_reasoner_parsed")
        except Exception:
            parsed = None
            print("[call_reasoner] Could not parse JSON from output (will return raw).")

    return raw_output, parsed
# ----------------------- Reasoner-assisted evaluation & remediation helpers -----------------------
def evaluate_answers_with_reasoner(lesson_json: Dict[str, Any], answers: Dict[str, str], sem_context: str, learner_age: Optional[int]) -> Dict[str, Any]:
    """
    Ask the reasoner to evaluate the student's answers against the lesson_json and the semantic context.
    Returns a dict with keys:
      - score: float (0..1)
      - details: [{id, expected, given, ok: bool, feedback: str}]
      - advice: short remedial advice string
    This function expects call_reasoner to return JSON-like output; falls back gracefully.
    """
    # construct compact payload
    # include explanation, quiz and student's answers and small context snippet
    quiz = lesson_json.get("quiz", []) or []
    compact_quiz = []
    for q in quiz:
        compact_quiz.append({
            "id": q.get("id"),
            "type": q.get("type"),
            "question": q.get("question"),
            "choices": q.get("choices"),
            "expected": q.get("correct_index") if q.get("correct_index") is not None else q.get("correct")
        })
    prompt = (
        "You are an expert grader and patient tutor. Evaluate the student's answers against the provided lesson and context.\n\n"
        "INPUT:\n"
        f"Lesson explanation (short): {json.dumps(lesson_json.get('explanation') or {}, ensure_ascii=False)[:3000]}\n\n"
        f"Quiz (compact): {json.dumps(compact_quiz, ensure_ascii=False)}\n\n"
        f"Student answers: {json.dumps(answers, ensure_ascii=False)}\n\n"
        f"Context snippet: {sem_context[:4000]}\n\n"
        f"Learner age: {learner_age}\n\n"
        "Return ONLY valid JSON with keys:\n"
        "- score: a number 0..1 representing fraction correct weighted by question type\n"
        "- details: list of objects {id: string, expected: string|null, given: string, ok: true|false, feedback: short string explaining the correct answer}\n"
        "- advice: short remedial advice (one paragraph) targeted to the learner age.\n\n"
        "Examples: {\"score\":0.67, \"details\":[...], \"advice\":\"...\"}\n\n"
        "Evaluate strictly but kindly. If student answer is an open text, check semantic correctness; give brief correction."
    )
    try:
        raw, parsed = call_reasoner(prompt, model=REASONING_MODEL, max_tokens=800, temperature=0.0, stream=False, try_extract_json=True)
    except Exception as e:
        debug_log_raw({"stage":"eval_reasoner_error","exc":str(e)}, tag="eval_reasoner_error")
        # fallback simple grading locally if reasoner unreachable
        details = []
        correct = 0
        for q in compact_quiz:
            qid = q.get("id")
            exp = q.get("expected")
            given = answers.get(qid, "")
            ok = False
            if exp is not None:
                try:
                    if str(exp).isdigit():
                        ok = (str(given).strip().upper() == chr(65 + int(exp)))
                    else:
                        ok = (str(given).strip().lower() == str(exp).strip().lower())
                except Exception:
                    ok = False
            details.append({"id": qid, "expected": exp, "given": given, "ok": bool(ok), "feedback": ""})
            if ok:
                correct += 1
        score = correct / max(1, len(compact_quiz))
        return {"score": score, "details": details, "advice": "Could not reach reasoner; local grading used."}

    # parse result
    eval_res = {}
    if isinstance(parsed, dict):
        eval_res = parsed
    else:
        # try robust extraction
        try:
            parsed2 = extract_json_from_text(raw or "")
            if isinstance(parsed2, dict):
                eval_res = parsed2
            else:
                # fallback simple: compute local
                eval_res = {"score": 0.0, "details": [], "advice": "Could not parse grader output."}
        except Exception:
            eval_res = {"score": 0.0, "details": [], "advice": "Could not parse grader output."}
    # ensure keys exist
    eval_res.setdefault("score", 0.0)
    eval_res.setdefault("details", [])
    eval_res.setdefault("advice", "")
    return eval_res


def produce_remedial_lesson(lesson_json: Dict[str, Any], sem_context: str, learner_age: Optional[int], simplify_level: int = 1) -> Dict[str, Any]:
    """
    Ask reasoner to produce a simpler explanation + short quiz targeted to learner_age.
    simplify_level increases simplification / scaffolding (1 = small, 2 = more).
    Returns a new lesson_json-like dict: {explanation:{title,bullets}, quiz:[...], remedial_instructions:[...]}
    """
    prompt = (
        "You are a patient tutor and instructional designer. Given the lesson explanation below and the context, "
        f"produce a SIMPLIFIED version tailored to a learner of age {learner_age}. Use simple language, short bullets, and clear examples.\n\n"
        "Also generate 3 short quiz questions (mcq or short open) with 'correct_index' for MCQs and provide short answer_explanations.\n\n"
        f"SIMPLIFY_LEVEL: {simplify_level}\n\n"
        f"EXPLANATION (original): {json.dumps(lesson_json.get('explanation') or {}, ensure_ascii=False)[:3000]}\n\n"
        f"CONTEXT_SNIPPET: {sem_context[:4000]}\n\n"
        "Return only JSON: {explanation:{title:..., bullets:[...]}, quiz:[{id,type,question,choices,correct_index,answer_explanation}], remedial_instructions:[...] }\n"
    )
    try:
        raw, parsed = call_reasoner(prompt, model=REASONING_MODEL, max_tokens=800, temperature=0.0, stream=False, try_extract_json=True)
    except Exception as e:
        debug_log_raw({"stage":"remedial_call_fail","exc":str(e)}, tag="remedial_call_fail")
        # fallback: return minimal simplified structure
        return {
            "explanation": {"title": lesson_json.get("explanation", {}).get("title", "Lesson"), "bullets": ["Simplified: review main points."]},
            "quiz": [],
            "remedial_instructions": ["Read the simplified bullets again."]
        }

    if isinstance(parsed, dict):
        return parsed
    else:
        try:
            parsed2 = extract_json_from_text(raw or "")
            if isinstance(parsed2, dict):
                return parsed2
        except Exception:
            pass
    # fallback minimal
    return {
        "explanation": {"title": lesson_json.get("explanation", {}).get("title", "Lesson"), "bullets": ["Simplified: review main points."]},
        "quiz": [],
        "remedial_instructions": ["Read the simplified bullets again."]
    }

# ------------------------- Higher-level helpers -------------------------
def ask_reasoner_for_keywords(query: str, num_keywords: int = 5) -> List[str]:
    """
    Ask the reasoner to return 3-10 short keywords (phrases).
    Returns a list of keywords (3..10).
    """
    n = max(3, min(10, int(num_keywords)))
    prompt = (
        "You are a concise assistant. Given the query below, return a JSON object with key 'keywords' "
        "whose value is a list of 3-10 short keyword phrases (no extra text). Example: {\"keywords\": [\"gravity\",\"laws of motion\"]}\n\n"
        f"Query: {query}\n\nReturn only the JSON."
    )

    print(f"[ask_reasoner_for_keywords] Requesting keywords for: {query}")
    debug_log_raw({"stage": "keywords_request", "query": query}, tag="keywords_request")

    raw, parsed = call_reasoner(prompt, model=REASONING_MODEL, max_tokens=200, temperature=0.2, stream=False, try_extract_json=True)
    kws = []
    if isinstance(parsed, dict):
        for key in ("keywords", "keyword", "key_phrases", "keys"):
            if parsed.get(key):
                v = parsed.get(key)
                if isinstance(v, list):
                    kws = [str(x).strip() for x in v if str(x).strip()]
                    break
                elif isinstance(v, str):
                    kws = [s.strip() for s in v.split(",") if s.strip()]
                    break

    if not kws:
        try:
            parsed2 = extract_json_from_text(raw)
            if isinstance(parsed2, dict):
                v = parsed2.get("keywords") or parsed2.get("keys") or parsed2.get("keyword")
                if isinstance(v, list):
                    kws = [str(x).strip() for x in v if str(x).strip()]
        except Exception:
            pass

    if not kws:
        first_line = (raw or "").strip().splitlines()[0] if raw else ""
        parts = [p.strip().strip('"').strip("'") for p in first_line.replace("keywords:", "").split(",") if p.strip()]
        kws = parts[:n]

    if not kws:
        kws = [w for w in query.split() if len(w) > 3][:n] or [query]

    # dedupe & limit
    out = []
    for k in kws:
        if k.lower() not in [x.lower() for x in out]:
            out.append(k)
        if len(out) >= n:
            break

    print(f"[ask_reasoner_for_keywords] Got keywords: {out}")
    debug_log_raw({"stage": "keywords_result", "keywords": out}, tag="keywords_result")
    return out

# keep earlier helpers though we will NOT use chunking
CHUNK_SIZE = 2000
CHUNK_SUMMARY_MAX_TOKENS = 600

def ask_reasoner_to_summarize_for_student(semantic_text: str, max_chars: int = 4000) -> Dict[str, Any]:
    """
    Given a chunk of semantic text, ask the reasoner to return pointers + summary.
    Returns {"pointers": [...], "summary": "..."}.
    (Kept for compatibility but not used in single-call pipeline.)
    """
    context = semantic_text[:max_chars]
    prompt = (
        "You are a patient tutor. Read the provided context and produce a JSON object ONLY with keys:\n"
        "- pointers: list of 12 clear, concise bullets (15-20 words each)\n"
        "DO NOT skip any content pointers in the context.\n"
        "- summary: one short paragraph (6 sentences)\n\n"
        "Return only the JSON object. No explanation or extra text.\n\n"
        f"CONTEXT:\n{context}"
    )

    raw, parsed = call_reasoner(prompt, max_tokens=CHUNK_SUMMARY_MAX_TOKENS, temperature=0.2, stream=False, try_extract_json=True)

    if isinstance(parsed, dict):
        pointers = parsed.get("pointers") or parsed.get("bullets") or parsed.get("points") or []
        summary = parsed.get("summary", "") or parsed.get("summary_text", "")
        return {"pointers": [str(p).strip() for p in pointers if str(p).strip()], "summary": str(summary).strip()}

    try:
        parsed2 = extract_json_from_text(raw)
        if isinstance(parsed2, dict):
            pointers = parsed2.get("pointers") or parsed2.get("bullets") or parsed2.get("points") or []
            summary = parsed2.get("summary", "") or parsed2.get("summary_text", "")
            return {"pointers": [str(p).strip() for p in pointers if str(p).strip()], "summary": str(summary).strip()}
    except Exception:
        pass

    lines = [ln.strip() for ln in (raw or "").splitlines() if ln.strip()]
    if not lines:
        lines = [ln.strip() for ln in context.splitlines() if ln.strip()]

    return {"pointers": lines[:8], "summary": (lines[0] if lines else "")}

def normalize_parsed_to_dict(parsed, raw_text=None):
    """
    Ensure we return a dict. If parsed is a dict, return as-is.
    If parsed is a list, try to pick a dict element or wrap.
    If parsed is scalar, attempt to extract JSON object from raw_text.
    """
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                return item
        return {"list_result": parsed}
    try:
        parsed2 = extract_json_from_text(raw_text or "")
        if isinstance(parsed2, dict):
            return parsed2
        if isinstance(parsed2, list):
            for it in parsed2:
                if isinstance(it, dict):
                    return it
            return {"list_result": parsed2}
    except Exception:
        pass
    return {"raw": raw_text or "", "note": "Could not parse structured lesson"}

# ------------------------ Helper: Run semantic_search_final.py per keyword -------------------------
def run_semantic_script_for_keyword(keyword: str, overall_progress_callback=None) -> Tuple[bool, str]:
    """
    Run the external semantic_search_final.py script for a given keyword.
    Writes the keyword to QUERY_PATH, runs the script with PYTHONUTF8 and PYTHONIOENCODING
    enabled for the subprocess, writes stdout/stderr logs, and appends any produced
    Temp_semantic.txt or stdout to REASONING_MODEL_TXT. On failure, stderr is appended
    so the pipeline preserves diagnostic information.
    """
    try:
        print(f"[run_semantic_script_for_keyword] Running semantic script for keyword: {keyword}")
        debug_log_raw({"stage": "run_semantic_start", "keyword": keyword}, tag="run_semantic_start")

        # 1) write keyword to query.txt
        try:
            with open(QUERY_PATH, "w", encoding="utf-8") as f:
                f.write(keyword)
        except Exception as e:
            msg = f"Failed to write query file {QUERY_PATH}: {e}"
            debug_log_raw(msg, tag="run_semantic_write_err")
            print("[run_semantic] Failed to write query file:", e)
            return False, msg

        # 2) ensure script exists
        if not SEMANTIC_SCRIPT.exists():
            msg = f"Semantic script not found: {SEMANTIC_SCRIPT}"
            debug_log_raw(msg, tag="run_semantic_not_found")
            print("[run_semantic] Script not found:", SEMANTIC_SCRIPT)
            return False, msg

        # prepare logs directory
        logs_dir = BASE_DIR / "semantic_logs"
        logs_dir.mkdir(exist_ok=True, parents=True)
        stdout_path = logs_dir / f"{keyword}_stdout.txt"
        stderr_path = logs_dir / f"{keyword}_stderr.txt"

        # prepare environment for subprocess: force utf-8 stdio to avoid UnicodeEncodeError
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        # run the script inside BASE_DIR
        try:
            proc = subprocess.run(
                [sys.executable, str(SEMANTIC_SCRIPT)],
                capture_output=True,
                text=True,
                encoding="utf-8",       # ensure parent decodes using UTF-8
                errors="replace",      # replace invalid bytes rather than raising
                cwd=str(BASE_DIR),
                timeout=300,
                env=env
            )
        except subprocess.TimeoutExpired as e:
            debug_log_raw(f"Script timeout for keyword={keyword}: {e}", tag="run_semantic_timeout")
            print(f"[run_semantic] Timeout for keyword {keyword}: {e}")
            return False, f"Timeout: {e}"
        except Exception as e:
            debug_log_raw(f"Script execution exception for keyword={keyword}: {e}", tag="run_semantic_exception")
            print(f"[run_semantic] Execution failed for {keyword}: {e}")
            return False, f"Execution failed: {e}"

        # persist stdout/stderr for later inspection
        try:
            stdout_path.write_text(proc.stdout or "", encoding="utf-8")
        except Exception:
            debug_log_raw(f"Failed to write stdout log for {keyword}", tag="write_stdout_err")
        try:
            stderr_path.write_text(proc.stderr or "", encoding="utf-8")
        except Exception:
            debug_log_raw(f"Failed to write stderr log for {keyword}", tag="write_stderr_err")

        # write truncated info to debug log
        debug_log_raw({
            "keyword": keyword,
            "returncode": proc.returncode,
            "stdout_head": (proc.stdout or "")[:4000],
            "stderr_head": (proc.stderr or "")[:4000]
        }, tag="run_semantic_output")

        # 3) prefer Temp_semantic.txt (FILE_PATH) if present and non-empty
        try:
            if FILE_PATH.exists():
                content = FILE_PATH.read_text(encoding="utf-8").strip()
                if content:
                    try:
                        with open(REASONING_MODEL_TXT, "a", encoding="utf-8") as f:
                            f.write(f"\n--- KEYWORD: {keyword} ---\n")
                            f.write(content + "\n")
                        print(f"[run_semantic] Appended Temp_semantic.txt content for {keyword}")
                    except Exception:
                        debug_log_raw(f"Failed to append Temp_semantic.txt content for {keyword}", tag="append_file_err")
                    return True, "Appended Temp_semantic.txt content"
        except Exception as e:
            debug_log_raw(f"Error reading {FILE_PATH}: {e}", tag="run_semantic_readfile_err")
            print(f"[run_semantic] Error reading Temp_semantic.txt: {e}")

        # 4) fallback: use stdout if it's non-empty
        out = (proc.stdout or "").strip()
        if out:
            try:
                with open(REASONING_MODEL_TXT, "a", encoding="utf-8") as f:
                    f.write(f"\n--- KEYWORD: {keyword} (from stdout) ---\n")
                    f.write(out + "\n")
                print(f"[run_semantic] Appended stdout for {keyword} to reasoningmodel.txt")
            except Exception:
                debug_log_raw(f"Failed to append stdout for {keyword}", tag="append_stdout_err")
            return True, "Appended stdout from script"

        # 5) nothing gathered: append stderr into reasoningmodel for traceability
        stderr_sample = (proc.stderr or "").strip()
        try:
            with open(REASONING_MODEL_TXT, "a", encoding="utf-8") as f:
                f.write(f"\n--- KEYWORD: {keyword} (stderr only) ---\n")
                f.write(stderr_sample + "\n")
            print(f"[run_semantic] No stdout; appended stderr for {keyword} to reasoningmodel.txt")
        except Exception:
            debug_log_raw(f"Failed to append stderr for {keyword}", tag="append_stderr_err")

        msg = f"Script ran with return code {proc.returncode} and produced only stderr. See {logs_dir / f'{keyword}_stderr.txt'}"
        debug_log_raw(msg, tag="run_semantic_no_stdout")
        return False, msg

    except Exception as e:
        debug_log_raw(f"Unhandled error running semantic script for keyword {keyword}: {e}\n{traceback.format_exc()}", tag="run_semantic_unhandled")
        print(f"[run_semantic] Unhandled error for {keyword}: {e}")
        return False, f"Unhandled: {e}"

# ------------------------- Progress / History helpers -------------------------
def load_progress_history() -> List[Dict[str, Any]]:
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            debug_log_raw(f"Failed to load progress file {PROGRESS_FILE}", tag="load_progress_err")
            return []
    return []

def save_progress_record(record: Dict[str, Any]) -> None:
    data = load_progress_history()
    data.append(record)
    try:
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception:
        debug_log_raw("Failed to save progress file", tag="save_progress_err")

def compute_mastery_for_module(module_name: str) -> float:
    """Compute rolling mastery for a module from history (0..1)."""
    data = load_progress_history()
    scores = [r.get("score") for r in data if r.get("module") == module_name and isinstance(r.get("score"), (int, float))]
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))

# ------------------------- Tkinter GUI -------------------------
class TutorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Interactive Tutor (Full)")
        self.geometry("1200x800")

        # top input area
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=6)

        ttk.Label(top, text="Module / Query:").pack(side="left")
        self.module_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.module_var, width=60).pack(side="left", padx=6)

        ttk.Label(top, text="Learner Age:").pack(side="left", padx=(10, 0))
        self.age_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.age_var, width=6).pack(side="left")

        ttk.Button(top, text="Start (Full flow)", command=self.start_full_flow).pack(side="left", padx=8)
        ttk.Button(top, text="Start (Single lesson)", command=self.start_lesson_once).pack(side="left", padx=8)
        ttk.Button(top, text="Clear reasoningmodel.txt", command=self.clear_reasoningmodel_file).pack(side="left", padx=8)

        # main area split: left = content, right = dashboard
        main = ttk.Frame(self)
        main.pack(fill="both", expand=True, padx=8, pady=6)

        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)

        # Context display
        ttk.Label(left, text="Context / Lesson / Quiz").pack(anchor="w")
        self.ctx_box = scrolledtext.ScrolledText(left, height=18, wrap="word")
        self.ctx_box.pack(fill="both", expand=True, padx=4, pady=4)

        # Question / doubt area (user can type a doubt and query the reasoner with current reasoningmodel.txt context)
        qframe = ttk.Frame(left)
        qframe.pack(fill="x", padx=4, pady=4)
        ttk.Label(qframe, text="Have a doubt? Type here and click Ask:").pack(anchor="w")
        self.doubt_text = tk.StringVar()
        ttk.Entry(qframe, textvariable=self.doubt_text, width=80).pack(side="left", padx=(0, 6))
        ttk.Button(qframe, text="Ask", command=self.ask_doubt).pack(side="left")

        # Concept / mastery progress bar
        pframe = ttk.Frame(left)
        pframe.pack(fill="x", padx=4, pady=4)
        ttk.Label(pframe, text="Concept Mastery:").pack(anchor="w")
        self.mastery_var = tk.DoubleVar(value=0.0)
        self.mastery_bar = ttk.Progressbar(pframe, variable=self.mastery_var, maximum=100)
        self.mastery_bar.pack(fill="x")

        # Bottom: last feedback
        ttk.Label(left, text="Tutor Explanation Output:").pack(anchor="w")
        self.output_box = scrolledtext.ScrolledText(left, height=10, wrap="word")
        self.output_box.pack(fill="both", expand=True, padx=4, pady=4)

        # Right sidebar: history + progress bars
        right = ttk.Frame(main, width=360)
        right.pack(side="left", fill="y", padx=6)
        ttk.Label(right, text="History / Dashboard").pack(anchor="w")
        self.history_list = tk.Listbox(right, height=20)
        self.history_list.pack(fill="y", expand=False, padx=4, pady=6)
        ttk.Button(right, text="Refresh History", command=self.refresh_history).pack(fill="x")

        # overall progress bar for keyword pipeline
        ttk.Label(right, text="Pipeline Progress:").pack(anchor="w", pady=(12, 0))
        self.pipeline_var = tk.DoubleVar(value=0.0)
        self.pipeline_bar = ttk.Progressbar(right, variable=self.pipeline_var, maximum=100)
        self.pipeline_bar.pack(fill="x", padx=4, pady=6)
        self.pipeline_status = ttk.Label(right, text="Idle")
        self.pipeline_status.pack(anchor="w", padx=4)

        # status bar
        self.status_label = ttk.Label(self, text="Ready", relief="sunken", anchor="w")
        self.status_label.pack(fill="x", side="bottom")

        # internal
        self.worker_thread: Optional[threading.Thread] = None
        self.task_queue: "queue.Queue" = queue.Queue()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # ensure reasoningmodel.txt exists
        REASONING_MODEL_TXT.parent.mkdir(parents=True, exist_ok=True)
        if not REASONING_MODEL_TXT.exists():
            REASONING_MODEL_TXT.write_text("", encoding="utf-8")

        self.refresh_history()
        self.after(500, self._process_task_queue)

    # -------------- UI helpers --------------
    def set_status(self, text: str) -> None:
        self.status_label.config(text=text)
        self.update_idletasks()

    def refresh_history(self) -> None:
        self.history_list.delete(0, tk.END)
        data = load_progress_history()
        for r in reversed(data[-200:]):
            timestr = datetime.fromtimestamp(r.get("timestamp", time.time())).isoformat()
            mod = r.get("module", "<no module>")
            score = r.get("score")
            line = f"{timestr} | {mod} | {score:.2%}" if isinstance(score, float) else f"{timestr} | {mod} | {score}"
            self.history_list.insert(tk.END, line)

    def clear_reasoningmodel_file(self) -> None:
        REASONING_MODEL_TXT.write_text("", encoding="utf-8")
        messagebox.showinfo("Cleared", f"{REASONING_MODEL_TXT} cleared.")

    # -------------- GUI actions --------------
    def ask_doubt(self) -> None:
        doubt = self.doubt_text.get().strip()
        if not doubt:
            messagebox.showwarning("No question", "Please type a doubt/question first.")
            return

        # read reasoningmodel.txt as context
        sem = REASONING_MODEL_TXT.read_text(encoding="utf-8")
        prompt = f"You are a helpful tutor. The student asks: {doubt}\n\nContext:\n{sem}\n\nAnswer clearly and simply."
        self.set_status("Sending doubt to reasoner...")
        self.output_box.insert("end", f"\n> Student question: {doubt}\n")
        self.update_idletasks()

        def _ask():
            try:
                raw, parsed = call_reasoner(prompt, model=REASONING_MODEL, max_tokens=400, temperature=0.2, stream=False, try_extract_json=True)
                if isinstance(parsed, dict):
                    s = json.dumps(parsed, indent=2, ensure_ascii=False)
                    out = f"Reasoner JSON response:\n{s}\n"
                else:
                    out = f"Reasoner response (raw):\n{raw}\n"
            except Exception as e:
                out = f"Error calling reasoner: {e}\n"
                logging.exception("ask_doubt error")
            self.task_queue.put(("doubt_result", out))

        threading.Thread(target=_ask, daemon=True).start()

    def start_lesson_once(self) -> None:
        """Ask reasoner for lesson using current Temp_semantic.txt (no keyword pipeline)."""
        module_query = self.module_var.get().strip()
        age = self.age_var.get().strip()
        if not module_query:
            messagebox.showwarning("Missing input", "Please enter a module/query.")
            return
        try:
            learner_age = int(age) if age else None
        except Exception:
            messagebox.showwarning("Invalid age", "Please enter numeric learner age.")
            return

        self.set_status("Generating lesson (single)...")
        self.pipeline_status.config(text="Generating lesson...")
        threading.Thread(target=self._worker_single_lesson, args=(module_query, learner_age), daemon=True).start()

    def start_full_flow(self) -> None:
        """Main flow: keywords -> run semantic script for each -> append to reasoningmodel.txt -> ask reasoner once to synthesize -> interactive quiz flow"""
        module_query = self.module_var.get().strip()
        age = self.age_var.get().strip()
        if not module_query:
            messagebox.showwarning("Missing input", "Please enter a module/query.")
            return
        try:
            learner_age = int(age) if age else None
        except Exception:
            messagebox.showwarning("Invalid age", "Please enter numeric learner age.")
            return

        # ensure previous reasoningmodel file preserved until pass threshold
        self.set_status("Starting full pipeline...")
        self.pipeline_status.config(text="Starting pipeline...")
        threading.Thread(target=self._worker_full_flow, args=(module_query, learner_age), daemon=True).start()

    # -------------- worker functions (background threads) --------------
    def _worker_single_lesson(self, module_query: str, learner_age: Optional[int]) -> None:
        try:
            # read Temp_semantic.txt as context
            sem = FILE_PATH.read_text(encoding="utf-8") if FILE_PATH.exists() else ""
            prompt = (
                "You are a patient tutor. Use the provided context to produce a JSON object ONLY with keys:\n"
                "explanation (title and bullets list), quiz (list of 3 questions with id,type,question,choices,correct), "
                "feedback (map id->explanation), progress_indicator (short string), remedial_instructions (list or string).\n\n"
                "Return only the JSON.\n\n"
                f"CONTEXT:\n{sem}\n\nMODULE: {module_query}\nLearner age: {learner_age}\nReturn only the JSON."
            )
            print("[_worker_single_lesson] Calling reasoner for single lesson; prompt_len:", len(prompt))
            debug_log_raw({"stage": "single_lesson_prompt", "len": len(prompt)}, tag="single_lesson_prompt")

            raw, parsed = call_reasoner(prompt, model=REASONING_MODEL, max_tokens=900, temperature=0.2, stream=False, try_extract_json=True)
            if isinstance(parsed, dict):
                lesson = parsed
            else:
                try:
                    lesson = extract_json_from_text(raw)
                except Exception as e:
                    debug_log_raw(f"Single lesson parse error: {e}\nRaw: {raw[:2000]}", tag="single_parse_err")
                    raise RuntimeError(f"Could not parse lesson JSON: {e}\nRaw:\n{raw[:2000]}")
        except Exception as e:
            logging.exception("Lesson generation error")
            self.task_queue.put(("lesson_error", str(e)))
            return
        self.task_queue.put(("lesson_ready", lesson))

    def _worker_full_flow(self, module_query: str, learner_age: Optional[int]) -> None:
        """
        Full pipeline worker — single-call pipeline (NO CHUNKING).
        Steps:
        1) Get keywords
        2) Run semantic_search_final.py for each keyword and append outputs to reasoningmodel.txt
        3) Read the entire reasoningmodel.txt and call reasoner ONCE to synthesize final lesson JSON
        4) Robustly save and queue the lesson, with verbose console + debug logging
        """
        try:
            # ENV / startup diagnostics
            try:
                cwd = os.getcwd()
            except Exception:
                cwd = "<cwd-unavailable>"
            print(f"[_worker_full_flow] START. CWD={cwd}  BASE_DIR={BASE_DIR}")
            debug_log_raw({"stage": "full_flow_start", "module_query": module_query, "cwd": cwd}, tag="full_flow_start")

            # 1) Get keywords
            print("[_worker_full_flow] Step 1: Requesting keywords...")
            self.task_queue.put(("pipeline_update", {"status": "Requesting keywords...", "percent": 0}))
            kws = ask_reasoner_for_keywords(module_query, num_keywords=6)
            if not kws:
                raise RuntimeError("Reasoner returned no keywords")
            total = len(kws)
            print(f"[_worker_full_flow] Got keywords ({total}): {kws}")
            self.task_queue.put(("pipeline_update", {"status": f"Got {total} keywords", "percent": 5}))
            debug_log_raw({"stage": "keywords_obtained", "keywords": kws}, tag="keywords_obtained")

            # 2) Truncate reasoningmodel file and run semantic script per keyword
            try:
                REASONING_MODEL_TXT.write_text("", encoding="utf-8")
                print(f"[_worker_full_flow] Cleared reasoning model file: {REASONING_MODEL_TXT}")
            except Exception as e:
                print("[_worker_full_flow] Warning: failed to clear reasoningmodel.txt:", e)
                debug_log_raw({"stage": "init_truncate_err", "exc": str(e)}, tag="init_truncate_err")

            print("[_worker_full_flow] Step 2: Running semantic script for each keyword")
            success_count = 0
            for i, kw in enumerate(kws, start=1):
                percent = int(5 + (i-1) * (80 / max(1, total)))
                status_msg = f"Processing keyword {i}/{total}: {kw}"
                self.task_queue.put(("pipeline_update", {"status": status_msg, "percent": percent}))
                print(f"[_worker_full_flow] {status_msg}")
                ok, msg = run_semantic_script_for_keyword(kw)
                if ok:
                    success_count += 1
                else:
                    print(f"[_worker_full_flow] Warning: semantic script failed for '{kw}': {msg}")
                    debug_log_raw({"keyword": kw, "msg": msg}, tag="keyword_failure")
                self.task_queue.put(("pipeline_update", {"status": f"Completed {i}/{total}", "percent": int(5 + i * (80 / max(1, total)))}))
                time.sleep(0.12)

            print(f"[_worker_full_flow] Step 2 complete: {success_count}/{total} succeeded")
            self.task_queue.put(("pipeline_update", {"status": f"Keyword pipeline done ({success_count}/{total})", "percent": 87}))
            debug_log_raw({"stage": "keyword_pipeline_done", "success": success_count, "total": total}, tag="keyword_pipeline_done")

            # 3) Read entire reasoningmodel.txt and prepare prompt
            print("[_worker_full_flow] Step 3: Reading entire reasoningmodel.txt")
            sem_text = REASONING_MODEL_TXT.read_text(encoding="utf-8")
            if not sem_text.strip():
                raise RuntimeError("No semantic text gathered (reasoningmodel.txt empty)")
            print(f"[_worker_full_flow] Read reasoningmodel.txt size={len(sem_text)} chars")
            debug_log_raw({"stage": "read_semantic_text", "len_chars": len(sem_text)}, tag="read_semantic_text")

            if len(sem_text) > MAX_PROMPT_CHARS:
                warn = f"reasoningmodel.txt is large ({len(sem_text)} chars). Truncating to {MAX_PROMPT_CHARS} chars for a single API call."
                print("[_worker_full_flow] WARNING:", warn)
                debug_log_raw(warn, tag="large_prompt_truncate")
                sem_to_send = sem_text[:MAX_PROMPT_CHARS]
            else:
                sem_to_send = sem_text

            self.task_queue.put(("pipeline_update", {"status": "Sending single summarization request (full context)...", "percent": 92}))

            final_prompt = (
                "You are a patient tutor and instructional designer. Using the context below (which contains gathered semantic search results), produce a JSON object ONLY with keys:\n"
                "  explanation: {title: string, bullets: [string,...]},\n"
                "  quiz: list of 3 questions where each question has {id: string, type: \"mcq\"|\"open\", question: string, choices: [..] or null, correct_index: int or null, answer_explanation: string},\n"
                "  feedback: map of question id -> short feedback string,\n"
                "  progress_indicator: short string describing expected mastery,\n"
                "  remedial_instructions: list of short actions for remediation.\n\n"
                "DO NOT include any extra text or markdown — return only valid JSON.\n\n"
                "CONTEXT:\n"
                f"{sem_to_send}\n\n"
                f"MODULE: {module_query}\nLearner_age: {learner_age}\n\nReturn only JSON."
            )

            print("[_worker_full_flow] Calling reasoner once for final lesson synthesis; prompt_len:", len(final_prompt))
            debug_log_raw({"stage": "final_synthesis_prompt", "len": len(final_prompt)}, tag="final_synthesis_prompt")

            raw, parsed = call_reasoner(final_prompt, model=None, max_tokens=1200, temperature=0.18, stream=False, try_extract_json=True)

            # ---------- Robust finalization & writes ----------
            print("[_worker_full_flow] Final call returned. Beginning finalization.")
            debug_log_raw({"stage": "final_call_raw_head", "raw_head": (raw or "")[:2000]}, tag="final_call_raw_head")

            # 1) Save the raw output (project dir)
            try:
                final_raw_path = BASE_DIR / "final_model_raw.txt"
                with open(final_raw_path, "w", encoding="utf-8") as rf:
                    rf.write(raw or "")
                print(f"[_worker_full_flow] Wrote raw model output -> {final_raw_path}")
                debug_log_raw({"stage":"wrote_raw","path":str(final_raw_path)}, tag="wrote_raw")
            except Exception as e:
                print("[_worker_full_flow] ERROR writing raw model output:", e)
                debug_log_raw({"stage":"write_raw_error","exc":str(e),"trace":traceback.format_exc()}, tag="write_raw_error")

            # 2) Parse / extract lesson JSON aggressively but forgivingly
            lesson_json_candidate = None
            parse_errors = []
            if isinstance(parsed, dict):
                lesson_json_candidate = parsed
                print("[_worker_full_flow] Parsed result is a dict (using parsed).")
            else:
                # try robust extractor
                try:
                    print("[_worker_full_flow] Trying extract_json_from_text(raw)...")
                    parsed2 = extract_json_from_text(raw or "")
                    if isinstance(parsed2, dict):
                        lesson_json_candidate = parsed2
                        print("[_worker_full_flow] extract_json_from_text -> dict")
                    elif isinstance(parsed2, list):
                        for it in parsed2:
                            if isinstance(it, dict):
                                lesson_json_candidate = it
                                print("[_worker_full_flow] extract_json_from_text -> list; using first dict element")
                                break
                except Exception as e:
                    parse_errors.append(str(e))
                    print("[_worker_full_flow] extract_json_from_text failed:", e)
                    debug_log_raw({"stage":"extract_exception","exc":str(e),"trace":traceback.format_exc()}, tag="extract_exception")

                # heuristic object substring
                if lesson_json_candidate is None:
                    try:
                        low = (raw or "").lower()
                        if '"explanation"' in low or '"quiz"' in low or '"pointers"' in low:
                            import re, json as _json
                            m = re.search(r'(\{(?:.|\n){50,}\})', raw or "", flags=re.MULTILINE)
                            if m:
                                cand = m.group(1)
                                try:
                                    obj = _json.loads(cand)
                                    if isinstance(obj, dict):
                                        lesson_json_candidate = obj
                                        print("[_worker_full_flow] Heuristic JSON substring parsed to dict.")
                                except Exception:
                                    pass
                    except Exception as e:
                        print("[_worker_full_flow] Heuristic extraction exception:", e)
                        debug_log_raw({"stage":"heuristic_exception","exc":str(e),"trace":traceback.format_exc()}, tag="heuristic_exception")

            # 3) fallback object if parse fails
            if lesson_json_candidate is None:
                lesson_json_candidate = {
                    "explanation": {"title": module_query or "Lesson", "bullets": ["(Fallback) See final_model_raw.txt for full output."]},
                    "quiz": [],
                    "feedback": {},
                    "progress_indicator": "Unknown",
                    "remedial_instructions": ["Inspect final_model_raw.txt"],
                    "raw_model_output_snippet": (raw or "")[:12000],
                    "parse_errors": parse_errors
                }
                print("[_worker_full_flow] Using fallback lesson_json_candidate (parsing failed).")
                debug_log_raw({"stage":"using_fallback","parse_errors":parse_errors}, tag="using_fallback")

            # 4) Attempt writes to several locations and report which succeeded
            wrote_targets = []
            # primary: project
            try:
                proj_path = BASE_DIR / "final_lesson_plan.json"
                try:
                    safe_write_text_file(proj_path, lesson_json_candidate, as_json=True)
                    wrote_targets.append(("project", str(proj_path)))
                    print(f"[_worker_full_flow] WROTE final_lesson_plan.json -> {proj_path} (project)")
                    debug_log_raw({"stage":"wrote_project","path":str(proj_path)}, tag="wrote_project")
                except Exception as e:
                    print("[_worker_full_flow] Failed to write to project path:", e)
                    debug_log_raw({"stage":"write_project_failed","exc":str(e),"trace":traceback.format_exc()}, tag="write_project_failed")

                wrote_targets.append(("project", str(proj_path)))
                print(f"[_worker_full_flow] WROTE final_lesson_plan.json -> {proj_path} (project)")
                debug_log_raw({"stage":"wrote_project","path":str(proj_path)}, tag="wrote_project")
            except Exception as e:
                print("[_worker_full_flow] Failed to write to project path:", e)
                debug_log_raw({"stage":"write_project_failed","exc":str(e),"trace":traceback.format_exc()}, tag="write_project_failed")

            # fallback: temp dir
            try:
                tempdir = Path(os.environ.get("TEMP") or os.environ.get("TMP") or "/tmp")
                temp_path = tempdir / "final_lesson_plan.json"
                try:
                    safe_write_text_file(proj_path, lesson_json_candidate, as_json=True)
                    wrote_targets.append(("project", str(proj_path)))
                    print(f"[_worker_full_flow] WROTE final_lesson_plan.json -> {proj_path} (project)")
                    debug_log_raw({"stage":"wrote_project","path":str(proj_path)}, tag="wrote_project")
                except Exception as e:
                    print("[_worker_full_flow] Failed to write to project path:", e)
                    debug_log_raw({"stage":"write_project_failed","exc":str(e),"trace":traceback.format_exc()}, tag="write_project_failed")

                wrote_targets.append(("temp", str(temp_path)))
                print(f"[_worker_full_flow] WROTE final_lesson_plan.json -> {temp_path} (temp)")
                debug_log_raw({"stage":"wrote_temp","path":str(temp_path)}, tag="wrote_temp")
            except Exception as e:
                print("[_worker_full_flow] Failed to write to temp path:", e)
                debug_log_raw({"stage":"write_temp_failed","exc":str(e),"trace":traceback.format_exc()}, tag="write_temp_failed")

            # fallback 2: Documents
            try:
                doc_path = Path.home() / "Documents" / "final_lesson_plan.json"
                doc_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    safe_write_text_file(proj_path, lesson_json_candidate, as_json=True)
                    wrote_targets.append(("project", str(proj_path)))
                    print(f"[_worker_full_flow] WROTE final_lesson_plan.json -> {proj_path} (project)")
                    debug_log_raw({"stage":"wrote_project","path":str(proj_path)}, tag="wrote_project")
                except Exception as e:
                    print("[_worker_full_flow] Failed to write to project path:", e)
                    debug_log_raw({"stage":"write_project_failed","exc":str(e),"trace":traceback.format_exc()}, tag="write_project_failed")

                wrote_targets.append(("documents", str(doc_path)))
                print(f"[_worker_full_flow] WROTE final_lesson_plan.json -> {doc_path} (Documents)")
                debug_log_raw({"stage":"wrote_documents","path":str(doc_path)}, tag="wrote_documents")
            except Exception as e:
                print("[_worker_full_flow] Failed to write to Documents path:", e)
                debug_log_raw({"stage":"write_documents_failed","exc":str(e),"trace":traceback.format_exc()}, tag="write_documents_failed")

            if wrote_targets:
                print("[_worker_full_flow] final_lesson_plan.json saved to:", wrote_targets)
            else:
                print("[_worker_full_flow] ERROR: final_lesson_plan.json was not written to any target. Check debug_model_raw.txt for details.")
                debug_log_raw({"stage":"write_none","note":"no targets succeeded"}, tag="write_none")

            # 5) queue lesson_ready for UI (always)
            try:
                self.task_queue.put(("lesson_ready", lesson_json_candidate))
                print("[_worker_full_flow] lesson_ready queued to UI.")
                debug_log_raw({"stage":"lesson_ready_queued","written":wrote_targets}, tag="lesson_ready_queued")
            except Exception as e:
                print("[_worker_full_flow] ERROR: failed to queue lesson_ready:", e)
                debug_log_raw({"stage":"queue_failed","exc":str(e),"trace":traceback.format_exc()}, tag="queue_failed")

            # 6) Launch viewer if available
            try:
                viewer_script = BASE_DIR / "serve_lesson.py"
                if viewer_script.exists():
                    print("[_worker_full_flow] Launching serve_lesson.py (detached).")
                    try:
                        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
                    except Exception:
                        creationflags = 0
                    if creationflags:
                        subprocess.Popen([sys.executable, str(viewer_script)], cwd=str(BASE_DIR), creationflags=creationflags)
                    else:
                        subprocess.Popen([sys.executable, str(viewer_script)], cwd=str(BASE_DIR))
                else:
                    print("[_worker_full_flow] serve_lesson.py not found; run it manually to view the lesson.")
            except Exception as e:
                print("[_worker_full_flow] Warning launching viewer:", e)
                debug_log_raw({"stage":"viewer_launch_failed","exc":str(e),"trace":traceback.format_exc()}, tag="viewer_launch_failed")

            print("[_worker_full_flow] Finalization completed.")

        except Exception as e:
            logging.exception("Full flow worker error")
            debug_log_raw({"stage": "full_flow_error", "error": str(e), "trace": traceback.format_exc()}, tag="full_flow_error")
            self.task_queue.put(("lesson_error", str(e)))
            self.task_queue.put(("pipeline_update", {"status": "Failed", "percent": 0}))


    # -------------- Task queue processor (runs in main thread via after) --------------
    def _process_task_queue(self):
        """
        Non-blocking queue processor. Keep work minimal here and schedule heavier UI actions via `after`.
        """
        try:
            while True:
                try:
                    item = self.task_queue.get_nowait()
                except queue.Empty:
                    break
                t = item[0]
                payload = item[1]
                # quick debug
                # print queue events to console for checkpointing
                print(f"[_process_task_queue] event: {t}")
                debug_log_raw({"event": t, "payload_preview": str(payload)[:100]}, tag="task_queue_event")

                if t == "doubt_result":
                    # small update, safe to apply directly
                    self.output_box.insert("end", payload + "\n")
                    self.set_status("Doubt answered")
                elif t == "pipeline_update":
                    self.pipeline_status.config(text=payload["status"])
                    self.pipeline_var.set(payload.get("percent", 0))
                elif t == "lesson_error":
                    self.set_status("Lesson failed; see debug log")
                    # show error on UI but avoid blocking main loop
                    self.output_box.insert("end", f"Lesson generation error:\n{payload}\n\n")
                    messagebox.showerror("Lesson error", str(payload))
                elif t == "lesson_ready":
                    # payload is lesson JSON (dict)
                    self.set_status("Lesson ready — rendering...")
                    # schedule display_lesson to run on the mainloop after a short delay so UI updates (progress bar) can repaint
                    # also log
                    debug_log_raw({"stage": "scheduling_display_lesson"}, tag="schedule_display")
                    self.after(50, lambda p=payload: self.display_lesson(p))
                else:
                    logging.warning("Unknown queue item: %s", t)
        finally:
            # reschedule quickly to keep UI responsive
            self.after(200, self._process_task_queue)


    # -------------- Display lesson & run quiz --------------
    def display_lesson(self, lesson_json: Dict[str, Any]) -> None:
        """
        Non-blocking lesson renderer: render explanation immediately, then schedule quiz via after().
        This prevents the UI from freezing while dialogs open and gives fast visual feedback.
        """
        try:
            if not isinstance(lesson_json, dict):
                self.output_box.insert("end", "Invalid lesson structure\n")
                self.set_status("Invalid lesson structure")
                return

            # quick checkpoint
            print("[display_lesson] Rendering explanation to UI (fast update).")
            debug_log_raw({"stage": "display_lesson_start", "keys": list(lesson_json.keys())}, tag="display_lesson_start")

            # show explanation quickly
            expl = lesson_json.get("explanation") or {}
            title = ""
            bullets = []
            # support two shapes: explanation can be dict or string
            if isinstance(expl, dict):
                title = expl.get("title", "")
                bullets = expl.get("bullets", []) or expl.get("points", []) or []
            elif isinstance(expl, str):
                # if explanation is plain text, use it as single bullet
                bullets = [expl]

            self.ctx_box.delete("1.0", "end")
            if title:
                self.ctx_box.insert("end", f"{title}\n\n")
            if bullets and isinstance(bullets, list):
                for b in bullets:
                    if isinstance(b, dict):
                        # support object bullets
                        t = b.get("title") or b.get("point") or ""
                        c = b.get("content") or b.get("text") or ""
                        if t:
                            line = f"• {t}: {c}\n"
                        else:
                            line = f"• {c}\n"
                        self.ctx_box.insert("end", line)
                    else:
                        self.ctx_box.insert("end", f"• {b}\n")
            self.ctx_box.see("1.0")
            self.output_box.insert("end", "Lesson explanation loaded.\n")
            self.output_box.see("end")

            # record that we've displayed explanation
            debug_log_raw({"stage": "explanation_rendered"}, tag="explanation_rendered")
            print("[display_lesson] Explanation displayed, scheduling quiz stage...")

            # schedule quiz logic to run slightly later so the UI has time to repaint
            self.after(100, lambda: self._launch_quiz_async(lesson_json))

        except Exception as e:
            logging.exception("display_lesson error")
            self.output_box.insert("end", f"Error displaying/processing lesson: {e}\n")
            self.set_status("Error during lesson display")
            debug_log_raw({"stage": "display_lesson_exception", "exc": str(e), "trace": traceback.format_exc()}, tag="display_lesson_exc")


    # grade quiz helper for the list structure
    def _grade_quiz_list(self, quiz: List[Dict[str, Any]], answers: Dict[str, str]) -> Tuple[float, List[Dict[str, Any]]]:
        correct = 0
        details = []
        for q in quiz:
            qid = q.get("id")
            typ = (q.get("type") or "mcq").lower()
            corr = q.get("correct")
            given = answers.get(qid, "")
            ok = False
            # normalize corr to string for comparison if it's not a boolean
            if isinstance(corr, bool):
                expected = "A" if corr else "B"
                ok = (str(given).strip().upper() == expected)
            else:
                expected = str(corr).strip()
                try:
                    if expected.isdigit():
                        g = str(given).strip().upper()
                        if g.isalpha():
                            idx = ord(g) - ord('A')
                            ok = (idx == int(expected))
                        else:
                            ok = (str(given).strip() == expected)
                    else:
                        ok = (str(given).strip().lower() == expected.lower())
                except Exception:
                    ok = (str(given).strip().lower() == expected.lower())
            details.append({"id": qid, "expected": expected, "given": given, "ok": bool(ok)})
            if ok:
                correct += 1
        score = float(correct) / max(1, len(quiz))
        return score, details
    
    def _launch_quiz_async(self, lesson_json: Dict[str, Any]) -> None:
        """
        Run interactive quiz + reasoner-validated evaluation and iterative remediation loop.
        Uses evaluate_answers_with_reasoner() and produce_remedial_lesson() helpers.
        """
        try:
            print("[_launch_quiz_async] Starting quiz-eval loop.")
            sys.stdout.flush()
            debug_log_raw({"stage": "_launch_quiz_async_start", "keys": list(lesson_json.keys())}, tag="_launch_quiz_async_start")

            # in-memory semantic context (use reasoningmodel.txt as source of truth)
            sem_context = REASONING_MODEL_TXT.read_text(encoding="utf-8")[:15000]

            current_lesson = lesson_json
            max_retries = 4
            attempt = 0
            while attempt < max_retries:
                attempt += 1
                self.output_box.insert("end", f"\n--- Attempt {attempt} — Quiz ---\n")
                self.output_box.see("end")

                quiz = current_lesson.get("quiz", []) or []
                if not quiz:
                    self.output_box.insert("end", "No quiz provided by model.\n")
                    self.set_status("Lesson ready (no quiz).")
                    return

                # collect answers (modal dialogs as before)
                answers = {}
                for q in quiz:
                    qid = q.get("id", str(time.time()))
                    qtype = (q.get("type") or "mcq").lower()
                    qtext = q.get("question") or ""
                    choices = q.get("choices") or []

                    if qtype == "mcq" and isinstance(choices, list) and len(choices) > 0:
                        prompt = qtext + "\n\n"
                        for i, ch in enumerate(choices):
                            opt = ch.get("title") if isinstance(ch, dict) else ch
                            prompt += f"  {chr(65+i)}. {opt}\n"
                        ans = self.simple_input_dialog("MCQ", prompt + "\nAnswer letter (A/B/...):")
                        answers[qid] = ans.strip() if ans else ""
                    else:
                        ans = self.simple_input_dialog("Answer", qtext)
                        answers[qid] = ans.strip() if ans else ""

                # send answers to reasoner for evaluation
                print(f"[_launch_quiz_async] Sending answers to reasoner for evaluation (attempt {attempt})")
                debug_log_raw({"stage":"eval_attempt","attempt":attempt,"answers_preview": {k: str(v)[:60] for k,v in answers.items()}}, tag="eval_attempt")
                eval_res = evaluate_answers_with_reasoner(current_lesson, answers, sem_context, int(self.age_var.get()) if self.age_var.get().isdigit() else None)

                # present evaluation
                score = float(eval_res.get("score", 0.0))
                details = eval_res.get("details", [])
                advice = eval_res.get("advice", "") or eval_res.get("explanation","")
                self.output_box.insert("end", f"\nEvaluation score: {score:.2%}\n")
                for d in details:
                    self.output_box.insert("end", json.dumps(d, ensure_ascii=False) + "\n")
                self.output_box.insert("end", f"Reasoner feedback / advice:\n{advice}\n")
                self.output_box.see("end")
                debug_log_raw({"stage":"eval_result","score":score,"details": details}, tag="eval_result")

                # store progress
                rec = {"module": self.module_var.get(), "timestamp": time.time(), "iteration": attempt, "score": float(score), "details": details}
                save_progress_record(rec)
                self.refresh_history()
                mastery = compute_mastery_for_module(self.module_var.get()) * 100.0
                self.mastery_var.set(mastery)

                # threshold check
                if score >= PASS_THRESHOLD:
                    self.output_box.insert("end", "\nCongratulations — passed threshold.\n")
                    self.set_status("Module passed.")
                    save_progress_record({"module": self.module_var.get(), "timestamp": time.time(), "score": float(score), "passed": True})
                    self.refresh_history()
                    return  # success: exit loop
                else:
                    # generate simplified remedial lesson + quiz and loop
                    self.output_box.insert("end", "\nScore below threshold. Generating remedial lesson and simpler quiz...\n")
                    self.output_box.see("end")
                    debug_log_raw({"stage":"need_remedial","attempt":attempt,"score":score}, tag="need_remedial")
                    # ask reasoner for simplified lesson (increase simplify level each attempt)
                    simplify_level = attempt  # 1..n -> slowly increase simplification
                    new_lesson = produce_remedial_lesson(current_lesson, sem_context, int(self.age_var.get()) if self.age_var.get().isdigit() else None, simplify_level=simplify_level)
                    # show simplified explanation quickly
                    expl = new_lesson.get("explanation", {})
                    title = expl.get("title","Remedial explanation")
                    bullets = expl.get("bullets",[]) or []
                    self.output_box.insert("end", f"\nRemedial explanation ({simplify_level}): {title}\n")
                    for b in bullets:
                        self.output_box.insert("end", f" - {b}\n")
                    self.output_box.see("end")
                    # set current_lesson to new_lesson and repeat
                    current_lesson = new_lesson
                    # small breather so UI updates
                    time.sleep(0.15)
                    continue

            # if loop exits without passing
            self.output_box.insert("end", "\nReached maximum remediation attempts. Please review the material manually or try a different topic.\n")
            self.set_status("Remediation attempts exhausted.")
            debug_log_raw({"stage":"remediation_exhausted","attempts":max_retries}, tag="remediation_exhausted")

        except Exception as e:
            logging.exception("Quiz/launch error")
            self.output_box.insert("end", f"Error during quiz loop: {e}\n")
            self.set_status("Quiz error")
            debug_log_raw({"stage":"quiz_error","exc": str(e), "trace": traceback.format_exc()}, tag="quiz_error")

    # simple input dialog reused
    def simple_input_dialog(self, title: str, prompt: str) -> str:
        # modal popup
        popup = tk.Toplevel(self)
        popup.title(title)
        popup.geometry("600x260")
        ttk.Label(popup, text=prompt, wraplength=560, justify="left").pack(padx=10, pady=10)
        entry_var = tk.StringVar()
        entry = ttk.Entry(popup, textvariable=entry_var, width=70)
        entry.pack(padx=10, pady=6)
        entry.focus_set()

        result = {"value": ""}

        def submit():
            result["value"] = entry_var.get()
            popup.destroy()

        ttk.Button(popup, text="Submit", command=submit).pack(pady=8)
        popup.transient(self)
        popup.grab_set()
        self.wait_window(popup)
        return result["value"]

    def on_close(self):
        # cleanup and close
        try:
            self.destroy()
        except Exception:
            pass

# ================ Main entrypoint ================
def main():
    # quick sanity check for model connectivity (non-blocking)
    try:
        raw, parsed = call_reasoner('Return JSON: {"keywords": ["gravity","mass","force"]}', try_extract_json=True)
        print("Reasoner sanity RAW sample:", (raw or "")[:400])
    except Exception as e:
        print("Warning: reasoner sanity check failed (model may be offline):", e)
        debug_log_raw(f"Sanity check failed: {e}", tag="sanity")

    app = TutorApp()
    app.mainloop()

if __name__ == "__main__":
    main()
