#!/usr/bin/env python3
"""
Total Recall — Conversation Browser

Browse and read conversations in the knowledge graph.

Usage:
  # List all sessions
  python3 browse.py

  # List sessions for a project
  python3 browse.py --project myproject

  # Read a specific session
  python3 browse.py --session <session_id>

  # Read with semantic links shown
  python3 browse.py --session <session_id> --show-links

  # Search and browse results in context
  python3 browse.py --search "agent evals"
"""

import sqlite3
import sys
import os
import argparse
from datetime import datetime

DEFAULT_DB = os.path.join(os.path.dirname(__file__), 'session_memory.db')


def open_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def list_sessions(conn, project=None):
    """List all sessions with summary info."""
    q = """
        SELECT s.session_id, s.project, s.message_count, s.source_file, s.ingested_at,
               MIN(c.timestamp) as first_msg, MAX(c.timestamp) as last_msg,
               COUNT(c.id) as chunk_count
        FROM sessions s
        LEFT JOIN chunks c ON c.session_id = s.session_id AND c.is_sidechain = 0
    """
    params = []
    if project:
        q += ' WHERE s.project = ?'
        params.append(project)
    q += ' GROUP BY s.session_id ORDER BY last_msg DESC'

    rows = conn.execute(q, params).fetchall()

    print(f'\n{"ID":12s}  {"Project":25s}  {"Chunks":>6s}  {"Date":12s}  Source')
    print('─' * 100)
    for r in rows:
        sid = r['session_id'][:12]
        proj = (r['project'] or 'global')[:25]
        chunks = r['chunk_count']
        date = (r['first_msg'] or '?')[:10]
        source = os.path.basename(r['source_file'] or '?')[:40]
        print(f'{sid}  {proj:25s}  {chunks:6d}  {date:12s}  {source}')

    print(f'\n{len(rows)} sessions')


def read_session(conn, session_id, show_links=False):
    """Read all messages in a session in order."""
    # Find the full session ID
    row = conn.execute(
        "SELECT session_id, project FROM sessions WHERE session_id LIKE ?",
        (session_id + '%',)
    ).fetchone()
    if not row:
        print(f'Session not found: {session_id}', file=sys.stderr)
        sys.exit(1)

    full_id = row['session_id']
    project = row['project']

    chunks = conn.execute("""
        SELECT * FROM chunks
        WHERE session_id = ? AND is_sidechain = 0
        ORDER BY seq_index
    """, (full_id,)).fetchall()

    print(f'\n══ Session {full_id[:12]} ══ Project: {project} ══ {len(chunks)} messages ══\n')

    for chunk in chunks:
        role = chunk['role'].upper()
        ts = (chunk['timestamp'] or '?')[:19]
        seq = chunk['seq_index']
        text = chunk['text'] or ''

        # Color coding
        if role == 'USER':
            header = f'\033[1;36m[{role} | #{seq} | {ts}]\033[0m'
        else:
            header = f'\033[1;33m[{role} | #{seq} | {ts}]\033[0m'

        print(header)

        # Wrap long lines
        for line in text.split('\n'):
            if len(line) > 120:
                words = line.split()
                current = ''
                for word in words:
                    if len(current) + len(word) + 1 > 120:
                        print(f'  {current}')
                        current = word
                    else:
                        current = f'{current} {word}' if current else word
                if current:
                    print(f'  {current}')
            else:
                print(f'  {line}')

        # Show semantic links if requested
        if show_links:
            links = conn.execute("""
                SELECT c.project, c.session_id, c.text, sl.similarity, sl.link_type
                FROM semantic_links sl
                JOIN chunks c ON c.id = sl.target_chunk_id
                WHERE sl.source_chunk_id = ?
                UNION
                SELECT c.project, c.session_id, c.text, sl.similarity, sl.link_type
                FROM semantic_links sl
                JOIN chunks c ON c.id = sl.source_chunk_id
                WHERE sl.target_chunk_id = ?
                ORDER BY similarity DESC
                LIMIT 3
            """, (chunk['id'], chunk['id'])).fetchall()

            if links:
                print(f'  \033[2m── semantic links ──\033[0m')
                for link in links:
                    ltype = link['link_type']
                    sim = link['similarity']
                    lproj = link['project']
                    ltext = (link['text'] or '')[:80].replace('\n', ' ')
                    print(f'  \033[2m  {ltype} (sim={sim:.3f}) [{lproj}] {ltext}\033[0m')

        print()


def search_and_browse(conn, query, project=None):
    """Search and show results with surrounding context."""
    from retrieve import retrieve, open_db as open_retrieve_db

    chunks = retrieve(
        query=query,
        db_path=conn.execute("PRAGMA database_list").fetchone()[2] or DEFAULT_DB,
        top_k=5,
        window=2,
        project=project,
    )

    if not chunks:
        print(f'No results for: {query}')
        return

    print(f'\n══ Search: "{query}" ══ {len(chunks)} results ══\n')

    prev_session = None
    for chunk in chunks:
        session = chunk['session_id'][:12]
        if session != prev_session:
            proj = chunk.get('project', '?')
            print(f'\033[1m── Session {session} [{proj}] ──\033[0m')
            prev_session = session

        role = chunk['role'].upper()
        ts = (chunk['timestamp'] or '?')[:10]
        seq = chunk['seq_index']
        text = (chunk['text'] or '')

        if len(text) > 300:
            text = text[:300] + '...'

        if role == 'USER':
            print(f'  \033[1;36m[{role} #{seq} {ts}]\033[0m')
        else:
            print(f'  \033[1;33m[{role} #{seq} {ts}]\033[0m')

        for line in text.split('\n')[:10]:
            print(f'    {line[:120]}')
        print()


def show_stats(conn):
    """Show knowledge graph statistics."""
    total_chunks = conn.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
    total_sessions = conn.execute('SELECT COUNT(*) FROM sessions').fetchone()[0]

    try:
        total_links = conn.execute('SELECT COUNT(*) FROM semantic_links').fetchone()[0]
    except:
        total_links = 0

    try:
        total_vecs = conn.execute('SELECT COUNT(*) FROM chunks_vec').fetchone()[0]
    except:
        total_vecs = 0

    total_embedded = conn.execute('SELECT COUNT(*) FROM chunk_embeddings').fetchone()[0]

    db_size = os.path.getsize(DEFAULT_DB) / 1024 / 1024

    print(f'\n══ Knowledge Graph Stats ══\n')
    print(f'  Chunks:          {total_chunks:,}')
    print(f'  Sessions:        {total_sessions:,}')
    print(f'  Embeddings:      {total_embedded:,}')
    print(f'  Vec index:       {total_vecs:,}')
    print(f'  Semantic links:  {total_links:,}')
    print(f'  DB size:         {db_size:.1f} MB')

    print(f'\n  By project:')
    for row in conn.execute(
        'SELECT project, COUNT(*) as cnt, COUNT(DISTINCT session_id) as sess '
        'FROM chunks GROUP BY project ORDER BY cnt DESC'
    ):
        print(f'    {row["project"]:25s}  {row["cnt"]:6,} chunks  {row["sess"]:3} sessions')

    if total_links > 0:
        print(f'\n  Link types:')
        for row in conn.execute(
            'SELECT link_type, COUNT(*) as cnt FROM semantic_links GROUP BY link_type ORDER BY cnt DESC'
        ):
            print(f'    {row["link_type"]:15s}  {row["cnt"]:,}')

    print()


def main():
    parser = argparse.ArgumentParser(description='Browse the session memory knowledge graph')
    parser.add_argument('--project', default=None, help='Filter by project')
    parser.add_argument('--session', default=None, help='Read a specific session')
    parser.add_argument('--search', default=None, help='Search and browse results')
    parser.add_argument('--show-links', action='store_true', help='Show semantic links per message')
    parser.add_argument('--stats', action='store_true', help='Show knowledge graph statistics')
    parser.add_argument('--db', default=DEFAULT_DB)
    args = parser.parse_args()

    conn = open_db(args.db)

    if args.stats:
        show_stats(conn)
    elif args.session:
        read_session(conn, args.session, show_links=args.show_links)
    elif args.search:
        search_and_browse(conn, args.search, project=args.project)
    else:
        list_sessions(conn, project=args.project)

    conn.close()


if __name__ == '__main__':
    main()
