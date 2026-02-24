from __future__ import annotations

from datetime import datetime, timedelta
import random

INTENTS = [
    ("kyc_doc_rejected", "onboarding", "kyc", "document rejected during KYC verification"),
    ("wire_inbound_delay", "funding", "transfers", "inbound wire transfer delayed"),
    ("2fa_phone_change", "security", "app_access", "2FA blocked after phone change"),
    ("order_rejected_margin", "trading", "orders", "order rejected due to insufficient buying power"),
    ("statement_missing", "reporting_tax", "statements", "monthly statement unavailable"),
    ("suspicious_login", "security", "security", "customer reported suspicious login notification"),
]
CHANNELS = ["app_chat", "web_chat", "email_like"]
LANGUAGES = ["pt", "en", "mixed"]
SEGMENTS = ["new", "active", "hv"]
RESOLUTION = ["resolved", "pending", "reopened"]


def generate_dataset(n: int, start_date: str, days: int, multi_intent_ratio: float, incident_spike_ratio: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    base_date = datetime.fromisoformat(start_date)
    rows: list[dict] = []
    for i in range(n):
        intent, journey_stage, product_area, problem = rng.choice(INTENTS)
        created_at = base_date + timedelta(days=rng.randint(0, max(days - 1, 0)), minutes=rng.randint(0, 1440))
        incident_flag = rng.random() < incident_spike_ratio
        second_intent = rng.choice(INTENTS)[0] if rng.random() < multi_intent_ratio else ""
        true_intent = intent if not second_intent else f"{intent}+{second_intent}"
        raw_transcript = f"Customer reports {problem}. Channel {rng.choice(CHANNELS)}. Lang {rng.choice(LANGUAGES)}."
        if second_intent:
            raw_transcript += f" Also asks about {second_intent}."

        rows.append(
            {
                "conversation_id": f"conv_{i}_{rng.getrandbits(32):08x}",
                "created_at": created_at.isoformat(),
                "channel": rng.choice(CHANNELS),
                "language_mix": rng.choice(LANGUAGES),
                "customer_segment": rng.choice(SEGMENTS),
                "journey_stage": journey_stage,
                "product_area": product_area,
                "resolution_status": rng.choices(RESOLUTION, weights=[72, 20, 8], k=1)[0],
                "incident_flag": incident_flag,
                "true_intent": true_intent,
                "synthetic_transcript_raw": raw_transcript,
            }
        )
    return rows
