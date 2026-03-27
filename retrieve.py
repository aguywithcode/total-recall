#!/usr/bin/env python3
"""
Total Recall — Retrieval Script
Hybrid FTS5 + sqlite-vec KNN search with RRF fusion, window expansion,
ancestor chain backtracking, and semantic link traversal.

Usage:
  python3 retrieve.py "your query here" [options]

Options:
  --db PATH        SQLite DB path (default: same dir as this script)
  --top-k N        Number of matches before expansion (default: 5)
  --window N       Context window ±N around each match (default: 2)
  --depth N        Ancestor chain depth (default: 3)
  --budget N       Max chunks in final output (default: 20)
  --tokens N       Max tokens in final output (default: 6000)
  --session ID     Restrict to a specific session_id
  --project NAME   Restrict to chunks tagged with this project
  --global         Search all projects (overrides --project)
  --format FORMAT  'context' (for Claude prompt) or 'json' (default: context)
  --roles ROLES    Comma-separated roles to include: user,assistant (default: both)
  --no-tools       Exclude pure tool_use/tool_result chunks
  --no-vectors     Disable vector search (FTS5 only)

Output (--format context):
  A formatted block of conversation excerpts ready to paste into a Claude prompt.

Output (--format json):
  JSON array of chunk objects for programmatic use.
"""

import json
import sys
import os
import argparse
import re
import struct

DEFAULT_DB = os.path.join(os.path.dirname(__file__), 'session_memory.db')

# ── DB CONNECTION ────────────────────────────────────────────────────────────

class DBConn:
    """Wrapper around apsw or sqlite3 connection to track capabilities."""
    def __init__(self, conn, has_vec=False, backend='sqlite3'):
        self._conn = conn
        self.has_vec = has_vec
        self.backend = backend

    def execute(self, sql, params=()):
        return self._conn.execute(sql, params)

    def close(self):
        return self._conn.close()


def open_db(db_path):
    """Open DB with apsw + sqlite-vec if available, fall back to sqlite3."""
    if not os.path.exists(db_path):
        print(f'ERROR: DB not found: {db_path}', file=sys.stderr)
        print('Run ingest.py first.', file=sys.stderr)
        sys.exit(1)

    try:
        import apsw
        import sqlite_vec
        conn = apsw.Connection(db_path, flags=apsw.SQLITE_OPEN_READONLY)
        conn.enable_load_extension(True)
        conn.load_extension(sqlite_vec.loadable_path())
        conn.enable_load_extension(False)
        return DBConn(conn, has_vec=True, backend='apsw')
    except ImportError:
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return DBConn(conn, has_vec=False, backend='sqlite3')


def _row_to_dict(row, conn):
    """Convert a row to a dict regardless of backend."""
    if conn.backend == 'apsw':
        # apsw returns tuples; we need column names from the cursor
        return row if isinstance(row, dict) else row
    else:
        return dict(row)


def _query(conn, sql, params=()):
    """Execute a query and return list of dicts."""
    cursor = conn.execute(sql, params)
    if conn.backend == 'apsw':
        desc = cursor.getdescription()
        col_names = [d[0] for d in desc]
        return [dict(zip(col_names, row)) for row in cursor.fetchall()]
    else:
        return [dict(row) for row in cursor.fetchall()]


def _query_one(conn, sql, params=()):
    rows = _query(conn, sql, params)
    return rows[0] if rows else None


# ── DB HELPERS ────────────────────────────────────────────────────────────────

def chunk_by_id(conn, chunk_id):
    return _query_one(conn, 'SELECT * FROM chunks WHERE id=?', (chunk_id,))


def chunks_by_seq_range(conn, seq_lo, seq_hi, session_id=None, roles=None, project=None):
    q = 'SELECT * FROM chunks WHERE seq_index >= ? AND seq_index <= ? AND is_sidechain=0'
    params = [seq_lo, seq_hi]
    if session_id:
        q += ' AND session_id=?'; params.append(session_id)
    if project:
        q += ' AND project=?'; params.append(project)
    if roles:
        placeholders = ','.join('?' * len(roles))
        q += f' AND role IN ({placeholders})'; params.extend(roles)
    q += ' ORDER BY seq_index'
    return _query(conn, q, params)


# ── FTS SEARCH ────────────────────────────────────────────────────────────────

def fts_search(conn, query, top_k, session_id=None, roles=None, no_tools=False, project=None):
    """Full-text search using FTS5 porter-stemmed index."""
    words = re.sub(r'[^\w\s]', ' ', query).split()
    if not words:
        return []
    fts_query = ' OR '.join(words)

    q = """
        SELECT c.*
        FROM chunks_fts f
        JOIN chunks c ON c.id = f.id
        WHERE chunks_fts MATCH ?
          AND c.is_sidechain = 0
    """
    params = [fts_query]
    if session_id:
        q += ' AND c.session_id=?'; params.append(session_id)
    if project:
        q += ' AND c.project=?'; params.append(project)
    if roles:
        placeholders = ','.join('?' * len(roles))
        q += f' AND c.role IN ({placeholders})'; params.extend(roles)
    if no_tools:
        q += " AND c.content_type NOT IN ('tool_use', 'tool_result')"
    q += ' ORDER BY rank LIMIT ?'
    params.append(top_k)

    return _query(conn, q, params)


# ── VECTOR SEARCH (sqlite-vec KNN) ──────────────────────────────────────────

def vector_search(conn, query_text, top_k, session_id=None, roles=None,
                  no_tools=False, project=None):
    """
    Vector search using sqlite-vec KNN index (fast) with brute-force fallback.
    Returns list of (similarity, chunk_dict) tuples, descending by similarity.
    """
    import embed
    query_vec = embed.get_embedding(query_text)

    if conn.has_vec:
        return _vec_knn_search(conn, query_vec, top_k, session_id, roles, no_tools, project)
    else:
        return _brute_force_search(conn, query_vec, top_k, session_id, roles, no_tools, project)


def _vec_knn_search(conn, query_vec, top_k, session_id=None, roles=None,
                    no_tools=False, project=None):
    """Fast KNN search via sqlite-vec."""
    vec_str = '[' + ','.join(str(f) for f in query_vec) + ']'
    fetch_k = top_k * 10  # overfetch to account for filters

    knn_rows = _query(conn, """
        SELECT chunk_id, distance
        FROM chunks_vec
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT ?
    """, (vec_str, fetch_k))

    results = []
    for knn in knn_rows:
        if len(results) >= top_k:
            break

        chunk = _query_one(conn, 'SELECT * FROM chunks WHERE id=?', (knn['chunk_id'],))
        if not chunk:
            continue

        if chunk.get('is_sidechain'):
            continue
        if project and chunk.get('project') != project:
            continue
        if session_id and chunk.get('session_id') != session_id:
            continue
        if roles and chunk.get('role') not in roles:
            continue
        if no_tools and chunk.get('content_type') in ('tool_use', 'tool_result'):
            continue

        # Use L2 distance as ranking (lower = more similar), convert to 0-1 score
        # Actual cosine similarity would require loading both vectors; L2 rank order
        # is sufficient for KNN since we just need relative ordering
        dist = knn['distance']
        similarity = 1.0 / (1.0 + dist)  # monotonic transform: smaller dist → higher sim
        results.append((similarity, chunk))

    return results


def _brute_force_search(conn, query_vec, top_k, session_id=None, roles=None,
                        no_tools=False, project=None):
    """Fallback brute-force cosine similarity (no sqlite-vec)."""
    import embed

    q = """
        SELECT ce.embedding, c.*
        FROM chunk_embeddings ce
        JOIN chunks c ON c.id = ce.chunk_id
        WHERE c.is_sidechain = 0
    """
    params = []
    if session_id:
        q += ' AND c.session_id=?'; params.append(session_id)
    if project:
        q += ' AND c.project=?'; params.append(project)
    if roles:
        placeholders = ','.join('?' * len(roles))
        q += f' AND c.role IN ({placeholders})'; params.extend(roles)
    if no_tools:
        q += " AND c.content_type NOT IN ('tool_use', 'tool_result')"

    rows = _query(conn, q, params)

    scored = []
    for row in rows:
        blob = row['embedding']
        n = len(blob) // 4
        chunk_vec = list(struct.unpack(f'<{n}f', blob))
        sim = embed.cosine_similarity(query_vec, chunk_vec)
        scored.append((sim, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


# ── RRF FUSION ────────────────────────────────────────────────────────────────

def rrf_merge(fts_results, vec_results, k=60):
    """Reciprocal Rank Fusion: merge two ranked lists."""
    scores = {}
    for rank, row in enumerate(fts_results):
        cid = row['id']
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

    for rank, (sim, row) in enumerate(vec_results):
        cid = row['id']
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── ANCESTOR BACKTRACKING ─────────────────────────────────────────────────────

def get_ancestors(conn, chunk_id, max_depth=3, hard_cap=5):
    """Walk the parentUuid chain upward from chunk_id."""
    ancestors = []
    current_id = chunk_id
    depth = 0

    while depth < max_depth and len(ancestors) < hard_cap:
        row = chunk_by_id(conn, current_id)
        if row is None:
            break
        parent_id = row['parent_id']
        if not parent_id:
            break

        parent = chunk_by_id(conn, parent_id)
        if parent is None:
            break

        if not parent['is_sidechain']:
            ancestors.append(parent)

        current_id = parent_id
        depth += 1

    ancestors.reverse()
    return ancestors


# ── SEMANTIC LINK TRAVERSAL ───────────────────────────────────────────────────

def get_semantic_links(conn, chunk_id, max_links=3, min_similarity=0.75):
    """Follow semantic back pointers to related chunks in other sessions/projects."""
    rows = _query(conn, """
        SELECT c.*, sl.similarity, sl.link_type
        FROM semantic_links sl
        JOIN chunks c ON c.id = sl.target_chunk_id
        WHERE sl.source_chunk_id = ?
          AND sl.similarity >= ?
          AND c.is_sidechain = 0
        ORDER BY sl.similarity DESC
        LIMIT ?
    """, (chunk_id, min_similarity, max_links))

    rows2 = _query(conn, """
        SELECT c.*, sl.similarity, sl.link_type
        FROM semantic_links sl
        JOIN chunks c ON c.id = sl.source_chunk_id
        WHERE sl.target_chunk_id = ?
          AND sl.similarity >= ?
          AND c.is_sidechain = 0
        ORDER BY sl.similarity DESC
        LIMIT ?
    """, (chunk_id, min_similarity, max_links))

    seen = set()
    merged = []
    for row in rows + rows2:
        if row['id'] not in seen:
            seen.add(row['id'])
            merged.append(row)

    merged.sort(key=lambda r: r['similarity'], reverse=True)
    return merged[:max_links]


# ── MAIN RETRIEVAL ────────────────────────────────────────────────────────────

def retrieve(query, db_path=DEFAULT_DB, top_k=5, window=2, depth=3,
             budget=20, token_budget=6000, session_id=None,
             roles=None, no_tools=False, no_vectors=False, project=None):
    """
    Hybrid retrieval pipeline:
    1a. FTS5 search → top_k keyword seeds
    1b. sqlite-vec KNN search → top_k semantic seeds (with brute-force fallback)
    1c. RRF fusion of both ranked lists
    2. For each seed: expand window ±N by seq_index
    3. For each seed: backtrack ancestor chain up to depth hops
    4. For each seed: traverse semantic links (cross-session back pointers)
    5. Deduplicate, sort by seq_index, apply budget cap
    6. Return ordered list of chunk dicts
    """
    conn = open_db(db_path)

    # 1a. FTS seeds
    fts_seeds = fts_search(conn, query, top_k, session_id, roles, no_tools, project)

    # 1b. Vector seeds
    vec_seeds = []
    if not no_vectors:
        try:
            vec_seeds = vector_search(conn, query, top_k, session_id, roles, no_tools, project)
        except Exception as e:
            print(f'<!-- vector search unavailable: {e} -->', file=sys.stderr)

    # 1c. Fuse with RRF or fall back to FTS-only
    if vec_seeds:
        ranked = rrf_merge(fts_seeds, vec_seeds)
        fused_ids = [cid for cid, score in ranked[:top_k]]
        all_rows = {r['id']: r for r in fts_seeds}
        all_rows.update({r['id']: r for _, r in vec_seeds})
        seeds = [all_rows[cid] for cid in fused_ids if cid in all_rows]
    else:
        seeds = fts_seeds

    if not seeds:
        conn.close()
        return []

    collected = {s['id']: s for s in seeds}

    # 2. Window expansion
    for seed in seeds:
        seq = seed['seq_index']
        if seq < 0:
            continue
        window_rows = chunks_by_seq_range(
            conn, max(0, seq - window), seq + window, session_id, roles, project
        )
        for row in window_rows:
            collected[row['id']] = row

    # 3. Ancestor backtracking
    for seed in seeds:
        ancestors = get_ancestors(conn, seed['id'], max_depth=depth)
        for row in ancestors:
            if row['id'] not in collected:
                collected[row['id']] = row

    # 4. Semantic link traversal (cross-session/cross-project back pointers)
    try:
        for seed in seeds:
            linked = get_semantic_links(conn, seed['id'], max_links=3)
            for row in linked:
                if row['id'] not in collected:
                    collected[row['id']] = row
    except Exception:
        pass  # table may not exist yet

    conn.close()

    # 5. Sort by seq_index, filter out negatives
    ordered = sorted(
        [r for r in collected.values() if r['seq_index'] >= 0],
        key=lambda r: r['seq_index']
    )

    # 6. Budget cap
    if len(ordered) > budget:
        seed_seqs = {s['seq_index'] for s in seeds if s['seq_index'] >= 0}
        priority = sorted(ordered, key=lambda r: (
            0 if r['seq_index'] in seed_seqs else 1,
            r['seq_index']
        ))
        kept_ids = {r['id'] for r in priority[:budget]}
        ordered = [r for r in ordered if r['id'] in kept_ids]

    # 7. Token budget
    result, total_tokens = [], 0
    for row in ordered:
        t = row['token_count'] or 0
        if total_tokens + t > token_budget and result:
            break
        result.append(row)
        total_tokens += t

    return result


# ── OUTPUT FORMATTERS ─────────────────────────────────────────────────────────

def format_context(chunks, query):
    """Format chunks as a Claude-ready context block."""
    lines = [
        f'<session_memory query="{query}">',
        f'<!-- {len(chunks)} excerpts retrieved from session history -->\n',
    ]
    prev_seq = None
    for chunk in chunks:
        seq  = chunk['seq_index']
        role = chunk['role'].upper()
        ts   = chunk['timestamp'][:10] if chunk['timestamp'] else '?'
        text = chunk['text'] or ''

        if prev_seq is not None and seq - prev_seq > 1:
            lines.append(f'\n[... {seq - prev_seq - 1} messages omitted ...]\n')

        if len(text) > 800:
            text = text[:780] + '\n… [truncated]'

        lines.append(f'[{role} | seq={seq} | {ts}]')
        lines.append(text)
        lines.append('')
        prev_seq = seq

    lines.append('</session_memory>')
    return '\n'.join(lines)


def format_json(chunks):
    out = []
    for c in chunks:
        out.append({
            'id':        c['id'],
            'seq_index': c['seq_index'],
            'role':      c['role'],
            'timestamp': c['timestamp'],
            'text':      c['text'],
        })
    return json.dumps(out, ensure_ascii=False, indent=2)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Retrieve relevant conversation excerpts from session memory'
    )
    parser.add_argument('query', help='Search query')
    parser.add_argument('--db',      default=DEFAULT_DB)
    parser.add_argument('--top-k',   type=int, default=5)
    parser.add_argument('--window',  type=int, default=2)
    parser.add_argument('--depth',   type=int, default=3)
    parser.add_argument('--budget',  type=int, default=20)
    parser.add_argument('--tokens',  type=int, default=6000)
    parser.add_argument('--session', default=None)
    parser.add_argument('--project', default=None,
                        help='Restrict to chunks tagged with this project')
    parser.add_argument('--global', dest='global_search', action='store_true',
                        help='Search all projects (overrides --project)')
    parser.add_argument('--format',  choices=['context', 'json'], default='context')
    parser.add_argument('--roles',   default=None,
                        help='Comma-separated: user,assistant')
    parser.add_argument('--no-tools', action='store_true')
    parser.add_argument('--no-vectors', action='store_true',
                        help='Disable vector search, use FTS5 only')
    args = parser.parse_args()

    roles = [r.strip() for r in args.roles.split(',')] if args.roles else None
    project = None if args.global_search else args.project

    chunks = retrieve(
        query       = args.query,
        db_path     = args.db,
        top_k       = args.top_k,
        window      = args.window,
        depth       = args.depth,
        budget      = args.budget,
        token_budget= args.tokens,
        session_id  = args.session,
        roles       = roles,
        no_tools    = args.no_tools,
        no_vectors  = args.no_vectors,
        project     = project,
    )

    if not chunks:
        print(f'No results for: {args.query}', file=sys.stderr)
        sys.exit(0)

    if args.format == 'json':
        print(format_json(chunks))
    else:
        print(format_context(chunks, args.query))


if __name__ == '__main__':
    main()
