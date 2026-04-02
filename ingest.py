#!/usr/bin/env python3
"""
Total Recall — Ingestion Pipeline
Reads Claude JSONL transcript → SQLite with FTS5 full-text search

Usage:
  python3 ingest.py <path/to/session.jsonl> [--db path/to/memory.db]

The database is created fresh (or updated incrementally) at --db location.
Existing chunks (matched by uuid) are skipped, so re-running is safe.
"""

import json
import sqlite3
import hashlib
import sys
import os
import argparse
from datetime import datetime
import embed

# ── CONFIG ──────────────────────────────────────────────────────────────────
DEFAULT_DB = os.path.join(os.path.dirname(__file__), 'session_memory.db')


# ── TEXT EXTRACTION ──────────────────────────────────────────────────────────
def extract_text(content, role):
    """Extract searchable text from a message content array."""
    if isinstance(content, str):
        return content.strip()

    parts = []
    for block in content:
        t = block.get('type', '')
        if t == 'text':
            text = block.get('text', '').strip()
            if text:
                parts.append(text)
        elif t == 'thinking':
            text = block.get('thinking', '').strip()
            if text:
                parts.append(f'[thinking] {text}')
        elif t == 'tool_use':
            name = block.get('name', '')
            inp  = block.get('input', {})
            # Summarise input (truncate large values)
            inp_str = json.dumps(inp, ensure_ascii=False)[:300]
            parts.append(f'[tool:{name}] {inp_str}')
        elif t == 'tool_result':
            result_content = block.get('content', '')
            if isinstance(result_content, list):
                result_text = ' '.join(
                    b.get('text', '') for b in result_content if b.get('type') == 'text'
                )
            else:
                result_text = str(result_content)
            parts.append(f'[result] {result_text[:500]}')

    return '\n'.join(parts).strip()


def content_type(content):
    if isinstance(content, str):
        return 'text'
    types = {b.get('type') for b in content if isinstance(b, dict)}
    if len(types) == 1:
        return list(types)[0]
    return 'mixed'


def estimate_tokens(text):
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


# ── SCHEMA ───────────────────────────────────────────────────────────────────
SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id    TEXT PRIMARY KEY,
    source_file   TEXT,
    ingested_at   TEXT,
    message_count INTEGER,
    project       TEXT DEFAULT 'global'
);

CREATE TABLE IF NOT EXISTS chunks (
    id            TEXT PRIMARY KEY,   -- message uuid
    session_id    TEXT,
    parent_id     TEXT,               -- parentUuid (NULL for root)
    is_sidechain  INTEGER DEFAULT 0,
    timestamp     TEXT,
    role          TEXT,               -- 'user' | 'assistant'
    content_type  TEXT,
    text          TEXT,               -- extracted searchable text
    token_count   INTEGER,
    content_hash  TEXT,               -- SHA-256 for file-move recovery
    source_file   TEXT,
    seq_index     INTEGER,            -- ordinal in linearised main thread (-1 = sidechain)
    project       TEXT DEFAULT 'global',
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_chunks_parent  ON chunks(parent_id);
CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunks(session_id);
CREATE INDEX IF NOT EXISTS idx_chunks_seq     ON chunks(seq_index);
CREATE INDEX IF NOT EXISTS idx_chunks_ts      ON chunks(timestamp);
CREATE INDEX IF NOT EXISTS idx_chunks_project ON chunks(project);

-- Full-text search over extracted text
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    id UNINDEXED,
    session_id UNINDEXED,
    role UNINDEXED,
    text,
    content=chunks,
    content_rowid=rowid,
    tokenize='porter unicode61'
);

-- Sync trigger: keep FTS in step with chunks table
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, id, session_id, role, text)
    VALUES (new.rowid, new.id, new.session_id, new.role, new.text);
END;

-- Vector embeddings (separate table to avoid FTS trigger issues)
CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id    TEXT PRIMARY KEY,
    model       TEXT,
    embedding   BLOB,
    created_at  TEXT,
    FOREIGN KEY (chunk_id) REFERENCES chunks(id)
);
"""


# ── LINEARISE MAIN THREAD ────────────────────────────────────────────────────
def linearise(messages):
    """
    Assign seq_index to main-thread messages in chronological order.
    Sidechain messages get seq_index = -1.

    Strategy: sort all non-sidechain messages by timestamp, assign
    seq_index 0, 1, 2, … This gives a stable linear ordering even when
    the parent chain has branches (e.g. tool round-trips).
    """
    main = [m for m in messages if not m.get('isSidechain', False)]
    main.sort(key=lambda m: m.get('timestamp', ''))
    seq_map = {m['uuid']: i for i, m in enumerate(main)}
    return seq_map


# ── INGEST ───────────────────────────────────────────────────────────────────
def migrate(conn):
    """Add project column to existing tables if missing."""
    for table in ('chunks', 'sessions'):
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN project TEXT DEFAULT 'global'")
        except Exception:
            pass  # column already exists
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_project ON chunks(project)")


def ingest(jsonl_path, db_path, no_embed=False, project='global'):
    print(f'  Source : {jsonl_path}')
    print(f'  DB     : {db_path}')

    # Read JSONL
    messages = []
    with open(jsonl_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Only process user/assistant messages (skip queue-operation etc.)
            if obj.get('type') not in ('user', 'assistant'):
                continue
            if 'uuid' not in obj:
                continue
            messages.append(obj)

    print(f'  Parsed : {len(messages)} user/assistant messages')

    # Determine session_id
    session_id = messages[0].get('sessionId', 'unknown') if messages else 'unknown'
    seq_map = linearise(messages)

    # Connect to DB
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    migrate(conn)

    # All-or-nothing: wrap session insert + chunk loop in a single transaction.
    # If the process is interrupted, the DB rolls back cleanly instead of
    # leaving partial chunks and a corrupted FTS index.
    try:
        conn.execute("BEGIN")

        # Register session
        conn.execute("""
            INSERT OR REPLACE INTO sessions(session_id, source_file, ingested_at, message_count, project)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, jsonl_path, datetime.utcnow().isoformat(), len(messages), project))

        # Ingest chunks
        inserted = 0
        skipped  = 0
        embedded = 0
        for msg in messages:
            uid = msg['uuid']

            # Skip if already exists
            if conn.execute('SELECT 1 FROM chunks WHERE id=?', (uid,)).fetchone():
                skipped += 1
                continue

            content = msg.get('message', {}).get('content', '')
            role    = msg.get('message', {}).get('role', msg.get('type', ''))
            text    = extract_text(content, role)
            ctype   = content_type(content) if isinstance(content, list) else 'text'
            chash   = hashlib.sha256(text.encode()).hexdigest()
            is_sc   = 1 if msg.get('isSidechain', False) else 0
            seq     = seq_map.get(uid, -1)

            conn.execute("""
                INSERT INTO chunks(id, session_id, parent_id, is_sidechain, timestamp,
                                   role, content_type, text, token_count, content_hash,
                                   source_file, seq_index, project)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                uid,
                session_id,
                msg.get('parentUuid'),
                is_sc,
                msg.get('timestamp', ''),
                role,
                ctype,
                text,
                estimate_tokens(text),
                chash,
                jsonl_path,
                seq,
                project,
            ))
            inserted += 1

            # Embed the chunk (skip short/empty text, respect --no-embed)
            if not no_embed and text and len(text) > 20:
                try:
                    vec = embed.get_embedding(text)
                    blob = embed.floats_to_blob(vec)
                    conn.execute("""
                        INSERT OR IGNORE INTO chunk_embeddings(chunk_id, model, embedding, created_at)
                        VALUES (?, ?, ?, ?)
                    """, (uid, embed.EMBED_MODEL, blob, datetime.utcnow().isoformat()))
                    embedded += 1
                except Exception as e:
                    print(f'  WARN: embedding failed for {uid[:8]}: {e}')

        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()

    total = inserted + skipped
    print(f'  Result : {inserted} inserted, {skipped} skipped ({total} total)')
    if not no_embed:
        print(f'  Embedded: {embedded} of {inserted} inserted chunks')
    print(f'  Done   : {db_path}')


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Ingest Claude JSONL into SQLite memory DB')
    parser.add_argument('jsonl', help='Path to .jsonl transcript file')
    parser.add_argument('--db', default=DEFAULT_DB, help='Path to SQLite DB (created if absent)')
    parser.add_argument('--no-embed', action='store_true', help='Skip embedding (FTS-only ingestion)')
    parser.add_argument('--project', default='global', help='Project tag for scoped retrieval (default: global)')
    args = parser.parse_args()

    if not os.path.exists(args.jsonl):
        print(f'ERROR: JSONL file not found: {args.jsonl}', file=sys.stderr)
        sys.exit(1)

    ingest(args.jsonl, args.db, no_embed=args.no_embed, project=args.project)


if __name__ == '__main__':
    main()
