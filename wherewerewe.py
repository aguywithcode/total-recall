#!/usr/bin/env python3
"""
Total Recall — Where Were We?

Shows the last N messages from the most recent session in a project.
Designed to be called at the start of a new session to quickly resume context.

Usage:
  python3 wherewerewe.py                          # last 20 from current project (default: global)
  python3 wherewerewe.py --project myproject         # last 20 from most recent session for that project
  python3 wherewerewe.py -n 50                     # last 50 messages
  python3 wherewerewe.py --all                     # last 20 across all projects (ignores project scope)
"""

import sqlite3
import sys
import os
import argparse

DEFAULT_DB = os.path.join(os.path.dirname(__file__), 'session_memory.db')


def where_were_we(db_path=DEFAULT_DB, project=None, n=20, all_projects=False):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Find the most recent session(s)
    if all_projects:
        # Most recent session across everything
        session = conn.execute("""
            SELECT session_id, project FROM chunks
            WHERE is_sidechain = 0
            ORDER BY timestamp DESC LIMIT 1
        """).fetchone()
    else:
        # Default to project scope, fall back to most recent
        p = project or 'global'
        session = conn.execute("""
            SELECT session_id, project FROM chunks
            WHERE project = ? AND is_sidechain = 0
            ORDER BY timestamp DESC LIMIT 1
        """, (p,)).fetchone()
        if not session:
            # Fallback to most recent across all
            session = conn.execute("""
                SELECT session_id, project FROM chunks
                WHERE is_sidechain = 0
                ORDER BY timestamp DESC LIMIT 1
            """).fetchone()

    if not session:
        print('No sessions found.', file=sys.stderr)
        sys.exit(0)

    session_id = session['session_id']
    proj = session['project']

    # Get the last N messages from this session
    chunks = conn.execute("""
        SELECT * FROM chunks
        WHERE session_id = ? AND is_sidechain = 0
        ORDER BY seq_index DESC
        LIMIT ?
    """, (session_id, n)).fetchall()

    # Reverse to chronological order
    chunks = list(reversed(chunks))

    # Get total message count for context
    total = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE session_id = ? AND is_sidechain = 0",
        (session_id,)
    ).fetchone()[0]

    conn.close()

    # Format output
    print(f'<session_context session="{session_id[:12]}" project="{proj}" showing_last="{len(chunks)}" of="{total}">')
    print()

    for chunk in chunks:
        role = chunk['role'].upper()
        ts = (chunk['timestamp'] or '?')[:19]
        seq = chunk['seq_index']
        text = chunk['text'] or ''

        # Truncate very long messages
        if len(text) > 600:
            text = text[:580] + '\n… [truncated]'

        print(f'[{role} | #{seq} | {ts}]')
        print(text)
        print()

    print('</session_context>')


def main():
    parser = argparse.ArgumentParser(
        description='Show the last N messages from the most recent session'
    )
    parser.add_argument('--project', '-p', default=None,
                        help='Project to look up (default: most recent across all)')
    parser.add_argument('-n', type=int, default=20,
                        help='Number of messages to show (default: 20)')
    parser.add_argument('--all', dest='all_projects', action='store_true',
                        help='Most recent session across all projects')
    parser.add_argument('--db', default=DEFAULT_DB)
    args = parser.parse_args()

    where_were_we(
        db_path=args.db,
        project=args.project,
        n=args.n,
        all_projects=args.all_projects,
    )


if __name__ == '__main__':
    main()
