#!/usr/bin/env python3
"""
Total Recall — Semantic Cross-Linker

Builds a semantic graph across all chunks by computing pairwise similarity
between chunks in different sessions/projects. Links are materialized in a
`semantic_links` table for traversal at retrieval time.

Runs as a background job after each ingest, or on-demand for the full corpus.

Usage:
  # Link new chunks (incremental — only chunks without outbound links)
  python3 semantic_linker.py --db session_memory.db

  # Relink everything (full rebuild)
  python3 semantic_linker.py --db session_memory.db --full

  # Link only a specific session's chunks
  python3 semantic_linker.py --db session_memory.db --session <session_id>

  # Link only a specific project's chunks
  python3 semantic_linker.py --db session_memory.db --project <project>
"""

import sqlite3
import struct
import sys
import os
import time
import argparse
from datetime import datetime, timezone

try:
    import embed
    HAS_EMBED = True
except Exception:
    HAS_EMBED = False

DEFAULT_DB = os.path.join(os.path.dirname(__file__), 'session_memory.db')

# ── SCHEMA ───────────────────────────────────────────────────────────────────

LINK_SCHEMA = """
CREATE TABLE IF NOT EXISTS semantic_links (
    source_chunk_id TEXT NOT NULL,
    target_chunk_id TEXT NOT NULL,
    similarity      FLOAT NOT NULL,
    link_type       TEXT DEFAULT 'related',
    created_at      TEXT,
    PRIMARY KEY (source_chunk_id, target_chunk_id),
    FOREIGN KEY (source_chunk_id) REFERENCES chunks(id),
    FOREIGN KEY (target_chunk_id) REFERENCES chunks(id)
);

CREATE INDEX IF NOT EXISTS idx_links_source ON semantic_links(source_chunk_id);
CREATE INDEX IF NOT EXISTS idx_links_target ON semantic_links(target_chunk_id);
CREATE INDEX IF NOT EXISTS idx_links_similarity ON semantic_links(similarity);
"""


def classify_link(source_row, target_row, similarity):
    """
    Classify the relationship between two chunks.

    - 'continuation': same topic, close in time, possibly across sessions
    - 'reference': high similarity but distant in time (callback to prior work)
    - 'related': moderate similarity, different context
    """
    # Parse timestamps
    try:
        s_time = datetime.fromisoformat(source_row['timestamp'].replace('Z', '+00:00'))
        t_time = datetime.fromisoformat(target_row['timestamp'].replace('Z', '+00:00'))
        time_delta = abs((s_time - t_time).total_seconds())
    except (ValueError, TypeError):
        time_delta = float('inf')

    same_project = source_row['project'] == target_row['project']

    # High similarity + close in time = continuation
    if similarity > 0.85 and time_delta < 86400:  # within 24 hours
        return 'continuation'

    # High similarity + distant time = reference/callback
    if similarity > 0.80 and time_delta > 86400 * 3:  # more than 3 days apart
        return 'reference'

    return 'related'


def _rows_to_dicts(cursor, keys=('id', 'session_id', 'project', 'timestamp', 'text', 'seq_index')):
    """Convert cursor results to list of dicts."""
    rows = cursor.fetchall()
    if not rows:
        return []
    if isinstance(rows[0], dict):
        return rows
    return [dict(zip(keys, row)) for row in rows]


def get_unlinked_chunks(conn, session_id=None, project=None):
    """Get chunks that have embeddings but no outbound semantic links yet."""
    q = """
        SELECT c.id, c.session_id, c.project, c.timestamp, c.text, c.seq_index
        FROM chunks c
        JOIN chunk_embeddings ce ON c.id = ce.chunk_id
        LEFT JOIN semantic_links sl ON c.id = sl.source_chunk_id
        WHERE sl.source_chunk_id IS NULL
          AND c.is_sidechain = 0
          AND c.text IS NOT NULL
          AND LENGTH(c.text) > 50
    """
    params = []
    if session_id:
        q += ' AND c.session_id = ?'
        params.append(session_id)
    if project:
        q += ' AND c.project = ?'
        params.append(project)
    q += ' ORDER BY c.seq_index'
    return _rows_to_dicts(conn.execute(q, params))


def get_all_linkable_chunks(conn, session_id=None, project=None):
    """Get all chunks with embeddings (for full rebuild)."""
    q = """
        SELECT c.id, c.session_id, c.project, c.timestamp, c.text, c.seq_index
        FROM chunks c
        JOIN chunk_embeddings ce ON c.id = ce.chunk_id
        WHERE c.is_sidechain = 0
          AND c.text IS NOT NULL
          AND LENGTH(c.text) > 50
    """
    params = []
    if session_id:
        q += ' AND c.session_id = ?'
        params.append(session_id)
    if project:
        q += ' AND c.project = ?'
        params.append(project)
    q += ' ORDER BY c.seq_index'
    return _rows_to_dicts(conn.execute(q, params))


def find_cross_session_neighbors(conn, chunk_id, chunk_session_id, top_k=5, threshold=0.75):
    """
    Find the top-K most similar chunks from OTHER sessions.
    Uses sqlite-vec KNN if available, falls back to brute-force.
    Returns list of (similarity, chunk_row) tuples.
    """
    # Check if chunks_vec table exists
    try:
        conn.execute("SELECT COUNT(*) FROM chunks_vec LIMIT 1").fetchone()
        return _vec_neighbors(conn, chunk_id, chunk_session_id, top_k, threshold)
    except Exception:
        return _brute_force_neighbors(conn, chunk_id, chunk_session_id, top_k, threshold)


def _vec_neighbors(conn, chunk_id, chunk_session_id, top_k, threshold):
    """Fast KNN neighbor search via sqlite-vec."""
    # Get this chunk's embedding as blob
    row = conn.execute(
        'SELECT embedding FROM chunk_embeddings WHERE chunk_id = ?', (chunk_id,)
    ).fetchone()
    if not row:
        return []

    # Convert blob to vec string
    blob = row[0] if isinstance(row, tuple) else row['embedding']
    n = len(blob) // 4
    source_vec = list(struct.unpack(f'<{n}f', blob))
    floats = source_vec
    vec_str = '[' + ','.join(str(f) for f in floats) + ']'

    # KNN search — overfetch to account for same-session filtering
    fetch_k = top_k * 10
    knn_rows = conn.execute("""
        SELECT chunk_id, distance
        FROM chunks_vec
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT ?
    """, (vec_str, fetch_k)).fetchall()

    results = []
    for knn_row in knn_rows:
        if len(results) >= top_k:
            break

        cid = knn_row[0] if isinstance(knn_row, tuple) else knn_row['chunk_id']
        dist = knn_row[1] if isinstance(knn_row, tuple) else knn_row['distance']

        # Get full chunk data
        chunk = conn.execute('SELECT * FROM chunks WHERE id = ?', (cid,)).fetchone()
        if not chunk:
            continue

        # Handle both tuple and Row results
        if isinstance(chunk, tuple):
            col_names = [d[1] for d in conn.execute('PRAGMA table_info(chunks)').fetchall()]
            chunk_dict = dict(zip(col_names, chunk))
        else:
            chunk_dict = dict(chunk)

        # Filter: different session, not sidechain, has text
        if chunk_dict.get('session_id') == chunk_session_id:
            continue
        if chunk_dict.get('is_sidechain'):
            continue
        if not chunk_dict.get('text') or len(chunk_dict.get('text', '')) <= 50:
            continue

        # Compute actual cosine similarity (L2→cosine approximation fails for unnormalized vecs)
        target_emb = conn.execute('SELECT embedding FROM chunk_embeddings WHERE chunk_id=?', (cid,)).fetchone()
        if not target_emb:
            continue
        t_blob = target_emb[0] if isinstance(target_emb, tuple) else target_emb['embedding']
        target_vec = list(struct.unpack(f'<{n}f', t_blob))
        similarity = embed.cosine_similarity(source_vec, target_vec)
        if similarity >= threshold:
            results.append((similarity, chunk_dict))

    return results


def _brute_force_neighbors(conn, chunk_id, chunk_session_id, top_k, threshold):
    """Fallback brute-force cosine similarity."""
    row = conn.execute(
        'SELECT embedding FROM chunk_embeddings WHERE chunk_id = ?', (chunk_id,)
    ).fetchone()
    if not row:
        return []

    blob = row[0] if isinstance(row, tuple) else row['embedding']
    source_vec = embed.blob_to_floats(blob)

    candidates = conn.execute("""
        SELECT ce.embedding, c.*
        FROM chunk_embeddings ce
        JOIN chunks c ON c.id = ce.chunk_id
        WHERE c.session_id != ?
          AND c.is_sidechain = 0
          AND c.text IS NOT NULL
          AND LENGTH(c.text) > 50
    """, (chunk_session_id,)).fetchall()

    scored = []
    for cand in candidates:
        cand_blob = cand[0] if isinstance(cand, tuple) else cand['embedding']
        cand_vec = embed.blob_to_floats(cand_blob)
        sim = embed.cosine_similarity(source_vec, cand_vec)
        if sim >= threshold:
            scored.append((sim, cand))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def link_chunks(db_path, full=False, session_id=None, project=None,
                top_k=5, threshold=0.75):
    """
    Build semantic links between chunks across sessions.

    For each source chunk, find the top-K most similar chunks from other
    sessions and create links if similarity exceeds threshold.
    """
    if not HAS_EMBED:
        print('ERROR: embed module not available. Cannot compute similarities.')
        sys.exit(1)

    # Try apsw + sqlite-vec for fast KNN, fall back to sqlite3
    has_vec = False
    try:
        import apsw
        import sqlite_vec
        conn = apsw.Connection(db_path)
        conn.enable_load_extension(True)
        conn.load_extension(sqlite_vec.loadable_path())
        conn.enable_load_extension(False)
        has_vec = True
        conn.execute("PRAGMA journal_mode=WAL")
        for stmt in LINK_SCHEMA.strip().split(';'):
            stmt = stmt.strip()
            if stmt:
                conn.execute(stmt)
    except ImportError:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(LINK_SCHEMA)

    if full:
        # Full rebuild — delete existing links for the scope
        if session_id:
            conn.execute('DELETE FROM semantic_links WHERE source_chunk_id IN '
                         '(SELECT id FROM chunks WHERE session_id = ?)', (session_id,))
        elif project:
            conn.execute('DELETE FROM semantic_links WHERE source_chunk_id IN '
                         '(SELECT id FROM chunks WHERE project = ?)', (project,))
        else:
            conn.execute('DELETE FROM semantic_links')
        try:
            conn.commit()
        except AttributeError:
            pass  # apsw autocommits
        sources = get_all_linkable_chunks(conn, session_id, project)
    else:
        sources = get_unlinked_chunks(conn, session_id, project)

    total = len(sources)
    if total == 0:
        print('All chunks already linked (or no chunks to link).')
        conn.close()
        return

    print(f'Linking {total} chunks (top_k={top_k}, threshold={threshold})...')
    start = time.time()

    linked = 0
    links_created = 0

    for i, source in enumerate(sources):
        neighbors = find_cross_session_neighbors(
            conn, source['id'], source['session_id'], top_k, threshold
        )

        for sim, target in neighbors:
            link_type = classify_link(source, target, sim)
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO semantic_links
                        (source_chunk_id, target_chunk_id, similarity, link_type, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    source['id'], target['id'], sim, link_type,
                    datetime.now(timezone.utc).isoformat()
                ))
                links_created += 1
            except Exception:
                pass

        linked += 1

        if (i + 1) % 50 == 0:
            try:
                conn.commit()
            except AttributeError:
                pass  # apsw autocommits
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            remaining = (total - i - 1) / rate if rate > 0 else 0
            print(f'  Progress: {i+1}/{total} ({links_created} links) '
                  f'— {rate:.1f} chunks/sec, ~{remaining:.0f}s remaining')

    try:
        conn.commit()
    except AttributeError:
        pass

    # Stats
    total_links = conn.execute('SELECT COUNT(*) FROM semantic_links').fetchone()[0]
    by_type = conn.execute(
        'SELECT link_type, COUNT(*) FROM semantic_links GROUP BY link_type'
    ).fetchall()

    conn.close()
    elapsed = time.time() - start

    print(f'\nDone: {linked} chunks processed, {links_created} new links — {elapsed:.1f}s')
    print(f'Total links in DB: {total_links}')
    for row in by_type:
        print(f'  {row[0]}: {row[1]}')


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Build semantic cross-links between conversation chunks'
    )
    parser.add_argument('--db', default=DEFAULT_DB)
    parser.add_argument('--full', action='store_true',
                        help='Full rebuild (delete and recompute all links)')
    parser.add_argument('--session', default=None,
                        help='Only link chunks from this session')
    parser.add_argument('--project', default=None,
                        help='Only link chunks from this project')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Max links per chunk (default: 5)')
    parser.add_argument('--threshold', type=float, default=0.75,
                        help='Minimum similarity to create link (default: 0.75)')
    args = parser.parse_args()

    link_chunks(
        db_path=args.db,
        full=args.full,
        session_id=args.session,
        project=args.project,
        top_k=args.top_k,
        threshold=args.threshold,
    )


if __name__ == '__main__':
    main()
