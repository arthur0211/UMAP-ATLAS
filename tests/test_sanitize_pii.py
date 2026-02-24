import re

import pytest

from umap_atlas.sanitize import sanitize_text, EMAIL_RE, PHONE_RE, DOC_RE


@pytest.mark.parametrize(
    "raw",
    [
        "Contact me at john.doe@mail.com phone +1-202-555-1212 doc 123456789",
        "Email: user+ops@sub.mail.com; tel (202) 555 1212; id 987654321",
        "meu email é ana.costa@invest.io e telefone +55 11 99888-7777 e doc 111222333",
        "docs 123456789 and 987654321 in one line",
    ],
)
def test_sanitize_removes_pii_patterns(raw: str) -> None:
    cleaned = sanitize_text(raw)
    assert EMAIL_RE.search(cleaned) is None
    assert PHONE_RE.search(cleaned) is None
    assert DOC_RE.search(cleaned) is None


def test_sanitize_idempotent() -> None:
    raw = "email foo@bar.com phone +1-202-555-9999 doc 123456789"
    once = sanitize_text(raw)
    twice = sanitize_text(once)
    assert once == twice


def test_sanitize_keeps_non_pii_text() -> None:
    raw = "unable to login after 2fa change in app"
    cleaned = sanitize_text(raw)
    assert re.sub(r"\s+", " ", cleaned).strip() == raw
