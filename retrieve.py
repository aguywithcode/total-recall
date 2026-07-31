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
  --explain        Print retrieval diagnostics to stderr (backend, seeds, timings)

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
import time

DEFAULT_DB = os.path.join(os.path.dirname(__file__), 'session_memory.db')

# Brute-force scans above this many seconds are worth complaining about.
SLOW_VECTOR_SECONDS = 1.5


# ── DIAGNOSTICS ──────────────────────────────────────────────────────────────

class VectorSearchUnavailable(RuntimeError):
    """Semantic search could not run; retrieval degrades to keyword-only."""


class RetrievalDiagnostics:
    """Records how a retrieval actually executed, including degraded paths.

    Vector search fails soft on purpose: if the embedding service or the
    embeddings themselves are missing, keyword results are still better than
    an error. The problem was that a degraded search looked identical to a
    healthy one, so silent quality loss could persist indefinitely. This type
    makes the degradation inspectable and, where it matters, noisy.
    """

    def __init__(self):
        self.backend = None            # 'apsw' | 'sqlite3'
        self.vector_mode = 'pending'   # knn | brute-force | disabled | unavailable
        self.vector_reason = None      # why degraded, when it is
        self.fts_seeds = 0
        self.vector_seeds = 0
        self.fused_seeds = 0
        self.embed_seconds = None
        self.vector_seconds = None
        self.scanned_embeddings = 0
        self.dim_mismatches = 0
        self.warnings = []

    def warn(self, message):
        """Record a user-visible warning, de-duplicated."""
        if message not in self.warnings:
            self.warnings.append(message)

    @property
    def degraded(self):
        """True when result *quality* was affected, not merely performance."""
        return self.vector_mode == 'unavailable' or bool(self.warnings)

    @property
    def suboptimal(self):
        """True when results are complete but the slow path was used."""
        return self.vector_mode == 'brute-force'

    def summary_lines(self):
        lines = [
            f'backend={self.backend} vector_mode={self.vector_mode}',
            f'seeds: fts={self.fts_seeds} vector={self.vector_seeds} '
            f'fused={self.fused_seeds}',
        ]
        if self.vector_reason:
            lines.append(f'vector_reason: {self.vector_reason}')
        if self.embed_seconds is not None:
            lines.append(f'embed_time={self.embed_seconds:.2f}s')
        if self.vector_seconds is not None:
            lines.append(f'vector_time={self.vector_seconds:.2f}s '
                         f'(scanned {self.scanned_embeddings} embeddings)')
        if self.dim_mismatches:
            lines.append(f'dimension_mismatches={self.dim_mismatches}')
        return lines

    def emit(self, stream=sys.stderr, verbose=False):
        """Write warnings (and optionally a full summary) as XML comments."""
        for message in self.warnings:
            print(f'<!-- {message} -->', file=stream)
        if verbose:
            for line in self.summary_lines():
                print(f'<!-- diag: {line} -->', file=stream)


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


def open_db(db_path, diag=None):
    """Open DB with apsw + sqlite-vec if available, fall back to sqlite3.

    Import failure and extension-load failure are handled separately: the
    former is the ordinary "not installed" case, while the latter (arch
    mismatch, sandbox restrictions, corrupt dylib) previously escaped as an
    unhandled exception even though a perfectly good fallback existed.
    """
    if not os.path.exists(db_path):
        print(f'ERROR: DB not found: {db_path}', file=sys.stderr)
        print('Run ingest.py first.', file=sys.stderr)
        sys.exit(1)

    try:
        import apsw
        import sqlite_vec
    except ImportError as exc:
        return _open_sqlite3(
            db_path, diag,
            f'sqlite-vec/apsw not installed ({exc}); using brute-force scan',
        )

    try:
        conn = apsw.Connection(db_path, flags=apsw.SQLITE_OPEN_READONLY)
        conn.enable_load_extension(True)
        conn.load_extension(sqlite_vec.loadable_path())
        conn.enable_load_extension(False)
    except Exception as exc:
        return _open_sqlite3(
            db_path, diag,
            f'sqlite-vec present but failed to load '
            f'({type(exc).__name__}: {exc}); using brute-force scan',
        )

    if diag is not None:
        diag.backend = 'apsw'
    return DBConn(conn, has_vec=True, backend='apsw')


def _open_sqlite3(db_path, diag, reason):
    """Open the stdlib sqlite3 fallback connection, recording why."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    if diag is not None:
        diag.backend = 'sqlite3'
        diag.vector_reason = reason
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
        first = next(cursor, None)
        if first is None:
            return []
        desc = cursor.getdescription()
        col_names = [d[0] for d in desc]
        results = [dict(zip(col_names, first))]
        results.extend(dict(zip(col_names, row)) for row in cursor)
        return results
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
                  no_tools=False, project=None, diag=None):
    """
    Vector search using sqlite-vec KNN index (fast) with brute-force fallback.
    Returns list of (similarity, chunk_dict) tuples, descending by similarity.

    Raises VectorSearchUnavailable when the query cannot be embedded, so the
    caller can degrade to keyword search with a specific, actionable reason
    rather than a bare exception string.
    """
    import embed

    started = time.perf_counter()
    try:
        query_vec = embed.get_embedding(query_text)
    except Exception as exc:
        raise VectorSearchUnavailable(_describe_embed_failure(exc, embed)) from exc
    if diag is not None:
        diag.embed_seconds = time.perf_counter() - started

    if conn.has_vec:
        if diag is not None:
            diag.vector_mode = 'knn'
        return _vec_knn_search(conn, query_vec, top_k, session_id, roles,
                               no_tools, project)

    if diag is not None:
        diag.vector_mode = 'brute-force'
    return _brute_force_search(conn, query_vec, top_k, session_id, roles,
                               no_tools, project, diag=diag)


def _describe_embed_failure(exc, embed_module):
    """Turn an embedding exception into a message that says what to do."""
    name = type(exc).__name__
    if isinstance(exc, ImportError):
        return (f'embedding dependency missing ({exc}) — '
                f'install it for this interpreter: {sys.executable}')
    if name in ('ConnectionError', 'ConnectTimeout', 'ReadTimeout', 'Timeout',
                'NewConnectionError', 'MaxRetryError'):
        url = getattr(embed_module, 'OLLAMA_URL', 'the embedding service')
        return (f'embedding service unreachable at {url} [{name}] — '
                f'is `ollama serve` running?')
    return f'{name}: {exc}'


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
                        no_tools=False, project=None, diag=None):
    """Fallback cosine-similarity scan for when sqlite-vec is unavailable.

    Two-phase by design. Phase 1 pulls only (id, embedding) for candidate
    chunks and ranks them; phase 2 hydrates full rows for the surviving top_k.
    The previous single-phase version selected `c.*` for every embedded chunk,
    materialising the entire corpus text just to score it, and carried the raw
    embedding blob downstream into the seed rows.
    """
    import embed

    q = """
        SELECT ce.chunk_id AS id, ce.embedding
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
    if diag is not None:
        diag.scanned_embeddings = len(rows)
    if not rows:
        return []

    expected_dim = len(query_vec)
    scored = []
    mismatches = 0

    for row in rows:
        blob = row['embedding']
        if not blob:
            continue
        n = len(blob) // 4
        # A dimension mismatch means the row was embedded with a different
        # model. Silently zipping vectors of unequal length would truncate to
        # the shorter one and yield a plausible-looking but meaningless score.
        if n != expected_dim:
            mismatches += 1
            continue
        chunk_vec = struct.unpack(f'<{n}f', blob)
        scored.append((embed.cosine_similarity(query_vec, chunk_vec), row['id']))

    if mismatches and diag is not None:
        diag.dim_mismatches = mismatches
        diag.warn(f'{mismatches} embedding(s) skipped: dimension != {expected_dim} '
                  f'(re-run backfill_embeddings.py to normalise)')

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    placeholders = ','.join('?' * len(top))
    hydrated = {
        row['id']: row
        for row in _query(conn,
                          f'SELECT * FROM chunks WHERE id IN ({placeholders})',
                          [cid for _, cid in top])
    }
    return [(sim, hydrated[cid]) for sim, cid in top if cid in hydrated]


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
             roles=None, no_tools=False, no_vectors=False, project=None,
             diag=None):
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

    Pass a RetrievalDiagnostics instance as `diag` to inspect how the search
    actually ran. When no instance is supplied, warnings are written to stderr
    so degradation is never entirely silent.
    """
    owns_diag = diag is None
    if owns_diag:
        diag = RetrievalDiagnostics()

    conn = open_db(db_path, diag=diag)

    # 1a. FTS seeds
    fts_seeds = fts_search(conn, query, top_k, session_id, roles, no_tools, project)
    diag.fts_seeds = len(fts_seeds)

    # 1b. Vector seeds
    vec_seeds = []
    if no_vectors:
        diag.vector_mode = 'disabled'
        diag.vector_reason = 'disabled by caller (--no-vectors)'
    else:
        started = time.perf_counter()
        try:
            vec_seeds = vector_search(conn, query, top_k, session_id, roles,
                                      no_tools, project, diag=diag)
        except VectorSearchUnavailable as exc:
            diag.vector_mode = 'unavailable'
            diag.vector_reason = str(exc)
            diag.warn(f'semantic search skipped, keyword results only — {exc}')
        except Exception as exc:
            # Not an expected degradation path; still fall back, but loudly.
            diag.vector_mode = 'unavailable'
            diag.vector_reason = f'unexpected {type(exc).__name__}: {exc}'
            diag.warn(f'semantic search failed unexpectedly, keyword results '
                      f'only — {type(exc).__name__}: {exc}')
        else:
            diag.vector_seconds = time.perf_counter() - started
            diag.vector_seeds = len(vec_seeds)
            if not vec_seeds:
                diag.warn('semantic search returned no candidates — the corpus '
                          'may have no embeddings yet (run backfill_embeddings.py)')
            elif (diag.vector_mode == 'brute-force'
                  and diag.vector_seconds > SLOW_VECTOR_SECONDS):
                diag.warn(f'brute-force vector scan took {diag.vector_seconds:.1f}s '
                          f'over {diag.scanned_embeddings} embeddings — install '
                          f'`apsw` and `sqlite-vec` for indexed KNN')

    # 1c. Fuse with RRF or fall back to FTS-only
    if vec_seeds:
        ranked = rrf_merge(fts_seeds, vec_seeds)
        fused_ids = [cid for cid, score in ranked[:top_k]]
        all_rows = {r['id']: r for r in fts_seeds}
        all_rows.update({r['id']: r for _, r in vec_seeds})
        seeds = [all_rows[cid] for cid in fused_ids if cid in all_rows]
    else:
        seeds = fts_seeds
    diag.fused_seeds = len(seeds)

    if not seeds:
        conn.close()
        if owns_diag:
            diag.emit()
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
    except Exception as exc:
        # Optional enrichment: the semantic_links table may not exist yet.
        # Recorded rather than swallowed, so a genuine schema problem is visible.
        diag.warn(f'semantic link traversal skipped '
                  f'({type(exc).__name__}: {exc})')

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

    if owns_diag:
        diag.emit()
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
    parser.add_argument('--explain', action='store_true',
                        help='Print retrieval diagnostics to stderr')
    args = parser.parse_args()

    roles = [r.strip() for r in args.roles.split(',')] if args.roles else None
    project = None if args.global_search else args.project

    diag = RetrievalDiagnostics()
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
        diag        = diag,
    )
    diag.emit(verbose=args.explain)

    if not chunks:
        print(f'No results for: {args.query}', file=sys.stderr)
        sys.exit(0)

    if args.format == 'json':
        print(format_json(chunks))
    else:
        print(format_context(chunks, args.query))


if __name__ == '__main__':
    main()
