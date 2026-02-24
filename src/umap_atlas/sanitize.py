from __future__ import annotations

import random
import re

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s\-]?)?(?:\(?\d{2,3}\)?[\s\-]?)\d{3,5}[\s\-]?\d{4}")
DOC_RE = re.compile(r"\b\d{9}\b")

FAKE_NAMES = ["Ana Costa", "John Miller", "Ravi Singh", "Maria Lopes"]
DOMAINS = ["mail.com", "invest.io", "bluepeak.test"]


def inject_fake_pii(rows: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    out: list[dict] = []
    for row in rows:
        new_row = row.copy()
        name = rng.choice(FAKE_NAMES)
        email = f"{name.lower().replace(' ', '.')}@{rng.choice(DOMAINS)}"
        phone = f"+1-202-{rng.randint(100,999)}-{rng.randint(1000,9999)}"
        doc = f"{rng.randint(100000000,999999999)}"
        new_row["synthetic_transcript_raw"] = (
            f"{new_row['synthetic_transcript_raw']} Customer {name}, email {email}, phone {phone}, document {doc}."
        )
        out.append(new_row)
    return out


def sanitize_text(text: str) -> str:
    txt = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    txt = PHONE_RE.sub("[REDACTED_PHONE]", txt)
    txt = DOC_RE.sub("[REDACTED_DOC]", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def sanitize_rows(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        new_row = row.copy()
        new_row["sanitized_transcript"] = sanitize_text(row["synthetic_transcript_raw"])
        out.append(new_row)
    return out
