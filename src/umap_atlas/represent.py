from __future__ import annotations


def build_embedding_text(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        new_row = row.copy()
        new_row["embedding_text"] = (
            f"Problema: {new_row['sanitized_transcript']} "
            f"Objetivo: resolver atendimento. "
            f"Area: {new_row['journey_stage']}/{new_row['product_area']}; "
            f"Canal: {new_row['channel']}; Idioma: {new_row['language_mix']}"
        )
        out.append(new_row)
    return out
