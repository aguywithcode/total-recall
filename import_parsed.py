#!/usr/bin/env python3
"""
Total Recall — Import Pre-Parsed Chat JSON

Imports conversations that have been manually parsed into structured JSON.
Expected format: JSON array of {role, text, timestamp, has_tool_use?} objects.

Usage:
  python3 import_parsed.py <path/to/parsed.json> --project <project> [--db path] [--no-embed]
"""

import json
import sqlite3
import hashlib
import uuid
import sys
import os
import argparse
from datetime import datetime, timezone

try:
    import embed
    HAS_EMBED = True
except Exception:
    HAS_EMBED = False

DEFAULT_DB = os.path.join(os.path.dirname(__file__), 'session_memory.db')


def estimate_tokens(text):
    return max(1, len(text) // 4)


def import_parsed(json_path, db_path, project, session_id=None, no_embed=False):
    with open(json_path, encoding='utf-8') as f:
        messages = json.load(f)

    if not messages:
        print('No messages in file.')
        return

    print(f'  Source  : {json_path}')
    print(f'  Project : {project}')
    print(f'  Messages: {len(messages)} ({sum(1 for m in messages if m["role"]=="user")} user, '
          f'{sum(1 for m in messages if m["role"]=="assistant")} assistant)')

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    from ingest import SCHEMA, migrate
    conn.executescript(SCHEMA)
    migrate(conn)

    if not session_id:
        session_id = str(uuid.uuid4())

    # Check if session already exists
    existing = conn.execute('SELECT 1 FROM sessions WHERE session_id=?', (session_id,)).fetchone()
    if existing:
        print(f'  Session {session_id[:8]} already exists, skipping.')
        conn.close()
        return

    conn.execute("""
        INSERT OR REPLACE INTO sessions(session_id, source_file, ingested_at, message_count, project)
        VALUES (?, ?, ?, ?, ?)
    """, (session_id, json_path, datetime.now(timezone.utc).isoformat(), len(messages), project))

    inserted = 0
    embedded = 0
    prev_uuid = None

    for i, msg in enumerate(messages):
        chunk_id = str(uuid.uuid4())
        text = msg['text']
        chash = hashlib.sha256(text.encode()).hexdigest()

        if conn.execute('SELECT 1 FROM chunks WHERE content_hash=? AND session_id=?',
                        (chash, session_id)).fetchone():
            prev_uuid = chunk_id
            continue

        content_type = 'mixed' if msg.get('has_tool_use') else 'text'

        conn.execute("""
            INSERT INTO chunks(id, session_id, parent_id, is_sidechain, timestamp,
                               role, content_type, text, token_count, content_hash,
                               source_file, seq_index, project)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            chunk_id, session_id, prev_uuid, 0,
            msg.get('timestamp', ''),
            msg['role'], content_type, text,
            estimate_tokens(text), chash,
            json_path, i, project,
        ))

        inserted += 1
        prev_uuid = chunk_id

        if not no_embed and HAS_EMBED and text and len(text) > 20:
            try:
                vec = embed.get_embedding(text)
                blob = embed.floats_to_blob(vec)
                conn.execute("""
                    INSERT OR IGNORE INTO chunk_embeddings(chunk_id, model, embedding, created_at)
                    VALUES (?, ?, ?, ?)
                """, (chunk_id, embed.EMBED_MODEL, blob, datetime.now(timezone.utc).isoformat()))
                embedded += 1
            except Exception:
                pass

        if (i + 1) % 20 == 0:
            conn.commit()

    conn.commit()
    conn.close()

    print(f'  Session : {session_id[:8]}')
    print(f'  Inserted: {inserted}')
    if not no_embed:
        print(f'  Embedded: {embedded}')
    print(f'  Done    : {db_path}')


def main():
    parser = argparse.ArgumentParser(description='Import pre-parsed chat JSON into session memory')
    parser.add_argument('file', help='Path to parsed JSON file')
    parser.add_argument('--project', required=True, help='Project tag')
    parser.add_argument('--db', default=DEFAULT_DB)
    parser.add_argument('--session-id', default=None)
    parser.add_argument('--no-embed', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f'ERROR: File not found: {args.file}', file=sys.stderr)
        sys.exit(1)

    import_parsed(args.file, args.db, args.project, args.session_id, args.no_embed)


if __name__ == '__main__':
    main()
