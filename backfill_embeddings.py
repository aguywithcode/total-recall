#!/usr/bin/env python3
"""
One-time backfill: embed all existing chunks that lack embeddings.
Resumable — safe to interrupt and re-run (skips already-embedded chunks).

Usage:
  python3 backfill_embeddings.py [--db path/to/memory.db]
"""

import sqlite3
import sys
import os
import time
from datetime import datetime, timezone
import embed

DEFAULT_DB = os.path.join(os.path.dirname(__file__), 'session_memory.db')


def backfill(db_path=DEFAULT_DB):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Use WAL mode for better write performance and less disk pressure
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    # Ensure embeddings table exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunk_embeddings (
            chunk_id    TEXT PRIMARY KEY,
            model       TEXT,
            embedding   BLOB,
            created_at  TEXT,
            FOREIGN KEY (chunk_id) REFERENCES chunks(id)
        )
    """)
    conn.commit()

    # Find chunks without embeddings, skip trivially short text
    rows = conn.execute("""
        SELECT c.id, c.text FROM chunks c
        LEFT JOIN chunk_embeddings ce ON c.id = ce.chunk_id
        WHERE ce.chunk_id IS NULL AND c.text IS NOT NULL AND LENGTH(c.text) > 20
        ORDER BY c.seq_index
    """).fetchall()

    total = len(rows)
    if total == 0:
        print("All chunks already embedded.")
        conn.close()
        return

    print(f"Backfilling {total} chunks...")
    start = time.time()

    done, failed = 0, 0
    for i, row in enumerate(rows):
        try:
            vec = embed.get_embedding(row['text'])
            blob = embed.floats_to_blob(vec)
            conn.execute(
                "INSERT OR IGNORE INTO chunk_embeddings(chunk_id, model, embedding, created_at) VALUES (?,?,?,?)",
                (row['id'], embed.EMBED_MODEL, blob, datetime.now(timezone.utc).isoformat()))
            done += 1
        except Exception as e:
            failed += 1
            print(f"  FAIL {row['id'][:8]}: {e}")

        if (i + 1) % 50 == 0:
            conn.commit()
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            remaining = (total - i - 1) / rate if rate > 0 else 0
            print(f"  Progress: {i+1}/{total} ({done} ok, {failed} fail) — {rate:.0f} chunks/sec, ~{remaining:.0f}s remaining")

    conn.commit()
    conn.close()
    elapsed = time.time() - start
    print(f"Done: {done} embedded, {failed} failed — {elapsed:.1f}s total")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Backfill embeddings for existing chunks')
    parser.add_argument('--db', default=DEFAULT_DB)
    args = parser.parse_args()
    backfill(args.db)
