#!/usr/bin/env python3
"""
Total Recall — sqlite-vec Integration

Provides a connection wrapper that loads sqlite-vec and manages the
vec0 virtual table for fast KNN vector search.

Uses apsw instead of stdlib sqlite3 because Python's sqlite3 module
doesn't support enable_load_extension on most builds.
"""

import apsw
import sqlite_vec
import struct
import os

DEFAULT_DB = os.path.join(os.path.dirname(__file__), 'session_memory.db')
EMBED_DIM = 768  # nomic-embed-text


def open_db(db_path=DEFAULT_DB):
    """Open a connection with sqlite-vec loaded."""
    conn = apsw.Connection(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(sqlite_vec.loadable_path())
    conn.enable_load_extension(False)
    return conn


def ensure_vec_table(conn):
    """Create the vec0 virtual table if it doesn't exist."""
    # Check if table exists
    exists = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='chunks_vec'"
    ).fetchone()[0]
    if not exists:
        conn.execute(f'CREATE VIRTUAL TABLE chunks_vec USING vec0(chunk_id TEXT PRIMARY KEY, embedding float[{EMBED_DIM}])')
    return not exists  # True if we just created it


def populate_vec_table(conn, batch_size=500):
    """
    Populate chunks_vec from existing chunk_embeddings table.
    Skips rows that already exist in chunks_vec.
    Returns count of rows inserted.
    """
    # Count what needs to be inserted
    total = conn.execute("""
        SELECT COUNT(*) FROM chunk_embeddings ce
        WHERE NOT EXISTS (SELECT 1 FROM chunks_vec cv WHERE cv.chunk_id = ce.chunk_id)
    """).fetchone()[0]

    if total == 0:
        return 0

    # Fetch and insert in batches
    inserted = 0
    offset = 0
    while offset < total:
        rows = conn.execute("""
            SELECT ce.chunk_id, ce.embedding
            FROM chunk_embeddings ce
            WHERE NOT EXISTS (SELECT 1 FROM chunks_vec cv WHERE cv.chunk_id = ce.chunk_id)
            LIMIT ? OFFSET ?
        """, (batch_size, offset)).fetchall()

        if not rows:
            break

        for chunk_id, blob in rows:
            # Convert BLOB to float list string for vec0
            n = len(blob) // 4
            floats = list(struct.unpack(f'<{n}f', blob))
            vec_str = '[' + ','.join(str(f) for f in floats) + ']'
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO chunks_vec(chunk_id, embedding) VALUES (?, ?)",
                    (chunk_id, vec_str)
                )
                inserted += 1
            except Exception:
                pass

        offset += batch_size
        if inserted % 1000 == 0 and inserted > 0:
            print(f'  Populated: {inserted}/{total}')

    return inserted


def knn_search(conn, query_embedding, top_k=5, project=None, session_id=None,
               roles=None, no_tools=False, exclude_session=None):
    """
    Fast KNN vector search using sqlite-vec.
    Returns list of (distance, chunk_row) tuples.
    """
    # Format the query vector
    vec_str = '[' + ','.join(str(f) for f in query_embedding) + ']'

    # KNN query against vec0 — returns closest by L2 distance
    # We fetch more than top_k to allow for filtering
    fetch_k = top_k * 5  # overfetch to account for filters

    knn_rows = conn.execute("""
        SELECT chunk_id, distance
        FROM chunks_vec
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT ?
    """, (vec_str, fetch_k)).fetchall()

    if not knn_rows:
        return []

    # Get full chunk data and apply filters
    results = []
    for chunk_id, distance in knn_rows:
        if len(results) >= top_k:
            break

        row = conn.execute('SELECT * FROM chunks WHERE id = ?', (chunk_id,)).fetchone()
        if not row:
            continue

        # Apply filters
        col_names = [d[0] for d in conn.execute('PRAGMA table_info(chunks)').fetchall()]
        row_dict = dict(zip(col_names, row))

        if row_dict.get('is_sidechain'):
            continue
        if project and row_dict.get('project') != project:
            continue
        if session_id and row_dict.get('session_id') != session_id:
            continue
        if exclude_session and row_dict.get('session_id') == exclude_session:
            continue
        if roles and row_dict.get('role') not in roles:
            continue
        if no_tools and row_dict.get('content_type') in ('tool_use', 'tool_result'):
            continue

        # Convert L2 distance to cosine similarity (approximate)
        # For normalized vectors: cosine_distance ≈ L2_distance² / 2
        similarity = max(0, 1 - (distance ** 2) / 2)
        results.append((similarity, row_dict))

    return results


def find_neighbors(conn, chunk_id, top_k=5, threshold=0.75, exclude_session=None):
    """
    Find the top-K most similar chunks to a given chunk.
    Uses sqlite-vec KNN for fast lookup.
    """
    # Get the source chunk's embedding from vec table
    row = conn.execute(
        "SELECT embedding FROM chunk_embeddings WHERE chunk_id = ?", (chunk_id,)
    ).fetchone()
    if not row:
        return []

    # Decode the blob to float list
    blob = row[0]
    n = len(blob) // 4
    floats = list(struct.unpack(f'<{n}f', blob))

    # Get the source chunk's session_id
    source = conn.execute('SELECT session_id FROM chunks WHERE id = ?', (chunk_id,)).fetchone()
    source_session = source[0] if source else None

    results = knn_search(
        conn, floats, top_k=top_k,
        exclude_session=exclude_session or source_session
    )

    # Filter by threshold
    return [(sim, row) for sim, row in results if sim >= threshold]


# ── CLI: Setup / Populate ────────────────────────────────────────────────────

def setup(db_path=DEFAULT_DB):
    """One-time setup: create vec table and populate from existing embeddings."""
    conn = open_db(db_path)
    created = ensure_vec_table(conn)
    if created:
        print(f'Created chunks_vec table (float[{EMBED_DIM}])')
    else:
        print('chunks_vec table already exists')

    print('Populating from existing embeddings...')
    count = populate_vec_table(conn)
    print(f'Inserted {count} vectors into chunks_vec')

    total = conn.execute('SELECT COUNT(*) FROM chunks_vec').fetchone()[0]
    print(f'Total vectors in index: {total}')
    conn.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Setup sqlite-vec index')
    parser.add_argument('--db', default=DEFAULT_DB)
    args = parser.parse_args()
    setup(args.db)
