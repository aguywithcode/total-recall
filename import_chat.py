#!/usr/bin/env python3
"""
Total Recall — Claude Chat Conversation Importer

Parses Claude.ai chat conversations (pasted as markdown/text) into the
session memory DB. Detects user/assistant turns, timestamps, and tool
usage. Creates a linear parent chain for ancestor backtracking.

Usage:
  python3 import_chat.py <path/to/conversation.md> --project <project> [--db path/to/memory.db]
  python3 import_chat.py <path/to/conversation.md> --project ddd-book --session-date 2026-03-18
  python3 import_chat.py <path/to/conversation.md> --project ddd-book --no-embed

The parser detects turn boundaries from patterns in Claude chat exports:
  - Date headers (Mar 18, Mar 19, etc.)
  - Time stamps (2:14 PM, 10:23 AM)
  - Tool usage indicators (Searched the web, Created a file, etc.)
  - Q&A formatted sections
  - File output markers (Document · MD, Document · DOCX)
"""

import re
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

# ── TURN DETECTION PATTERNS ─────────────────────────────────────────────────

# Date headers like "Mar 18", "Mar 19", "March 18, 2026"
DATE_HEADER = re.compile(
    r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2}(?:,?\s+\d{4})?$',
    re.MULTILINE
)

# Time stamps like "2:14 PM", "10:23 AM"
TIME_STAMP = re.compile(r'^\d{1,2}:\d{2}\s*[AP]M$', re.MULTILINE)

# Tool usage indicators (assistant-only)
TOOL_INDICATORS = [
    'Searched the web',
    'Created a file',
    'Edited a file',
    'Ran a command',
    'Ran ',
    'Viewed a file',
    'Reading the',
    'Fetched:',
    'Extract text',
    'Check ',
    'Find ',
    'Search ',
    'Update ',
    'Add ',
    'Replace ',
    'Rename ',
    'Count ',
    'Now let me',
    'Now update',
    'Now add',
    'Now fix',
]

# File output markers
FILE_OUTPUT = re.compile(r'^.+\nDocument · (MD|DOCX|PDF|TXT)\s*$', re.MULTILINE)

# Q&A pattern (user choosing from options)
QA_PATTERN = re.compile(r'^Q:\s+.+\nA:\s+.+$', re.MULTILINE)

# Send via Gmail / other action buttons
ACTION_BUTTON = re.compile(r'^Send via (Gmail|Slack|Email)', re.MULTILINE)

# Assistant continuation markers
ASSISTANT_MARKERS = [
    'Let me ',
    'Here\'s ',
    'Good ',
    'Great ',
    'Perfect',
    'That\'s ',
    'Done.',
    'Now ',
    'I\'ll ',
    'I can',
    'I have',
    'I found',
    'I wasn\'t',
    'This is ',
    'The ',
    'A few ',
    'Three ',
    'Two ',
    'For ',
    'On ',
    'Both ',
    'All ',
    'Yes',
    'No ',
    'Ha ',
    'Love ',
    'Exactly',
    'Right',
    'Agreed',
    'Subject:',
    'Commissioner',
    'Madison',
]


# ── PARSER ───────────────────────────────────────────────────────────────────

def parse_conversation(text, session_date=None, year=2026):
    """
    Parse a Claude chat conversation into a list of message dicts.

    Returns list of:
    {
        'role': 'user' | 'assistant',
        'text': str,
        'timestamp': str (ISO format),
        'has_tool_use': bool,
    }
    """
    messages = []
    current_date = session_date or f'{year}-03-18'
    current_role = None
    current_text = []
    current_has_tools = False

    def flush():
        nonlocal current_role, current_text, current_has_tools
        if current_role and current_text:
            text_joined = '\n'.join(current_text).strip()
            if text_joined and len(text_joined) > 5:  # skip trivially short
                messages.append({
                    'role': current_role,
                    'text': text_joined,
                    'timestamp': current_date + 'T12:00:00.000Z',
                    'has_tool_use': current_has_tools,
                })
        current_text = []
        current_has_tools = False

    lines = text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        # Skip empty lines at turn boundaries
        if not line.strip():
            if current_text:
                current_text.append('')
            i += 1
            continue

        # Date header — update current date
        date_match = DATE_HEADER.match(line.strip())
        if date_match:
            parsed_date = _parse_date_header(line.strip(), year)
            if parsed_date:
                current_date = parsed_date
            i += 1
            continue

        # Time stamp — signals a new turn
        time_match = TIME_STAMP.match(line.strip())
        if time_match:
            # Next non-empty line determines who's speaking
            i += 1
            continue

        # Tool usage line — this is assistant, may start new turn or continue
        is_tool_line = any(line.strip().startswith(marker) for marker in TOOL_INDICATORS)
        if is_tool_line:
            if current_role != 'assistant':
                flush()
                current_role = 'assistant'
            current_has_tools = True
            current_text.append(line)
            i += 1
            continue

        # File output marker
        if FILE_OUTPUT.match(line):
            if current_role == 'assistant':
                current_text.append(line)
            i += 1
            continue

        # Action button (Send via Gmail etc.)
        if ACTION_BUTTON.match(line.strip()):
            if current_role == 'assistant':
                current_text.append(line)
            i += 1
            continue

        # Q&A pattern — user answering a question
        qa_match = QA_PATTERN.match(line)
        if qa_match:
            flush()
            current_role = 'user'
            current_text.append(line)
            i += 1
            continue

        # Detect role from content heuristics
        detected_role = _detect_role(line, current_role)

        if detected_role and detected_role != current_role:
            flush()
            current_role = detected_role

        if current_role is None:
            # First message — guess from content
            current_role = _detect_role(line, None) or 'user'

        current_text.append(line)
        i += 1

    flush()

    # Assign sequential timestamps
    _assign_timestamps(messages, year)

    return messages


def _parse_date_header(text, year):
    """Parse 'Mar 18' or 'March 18, 2026' into ISO date string."""
    months = {
        'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
        'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
        'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12',
    }
    parts = text.replace(',', '').split()
    if len(parts) >= 2:
        month_key = parts[0][:3].lower()
        if month_key in months:
            day = parts[1].zfill(2)
            y = parts[2] if len(parts) > 2 else str(year)
            return f'{y}-{months[month_key]}-{day}'
    return None


def _detect_role(line, current_role):
    """
    Heuristically detect whether a line is from user or assistant.
    Returns 'user', 'assistant', or None (continue current role).
    """
    stripped = line.strip()
    if not stripped:
        return None

    # Strong user signals
    if stripped.startswith('Q:') or stripped.startswith('A:'):
        return 'user'

    # User patterns: short, directive, questions, file references
    if len(stripped) < 150:
        # Short imperatives/questions are usually user
        user_starters = [
            'yes', 'no', 'sure', 'let\'s', 'can you', 'can we',
            'do you', 'what', 'how', 'where', 'when', 'why',
            'I ', 'i ', 'we ', 'my ', 'here', 'ok', 'okay',
            'fixed', 'submitted', 'done', 'I\'m ', 'i\'m ',
            'also', 'one note', 'remember', 'don\'t', 'hint',
            'assuming', 'perhaps', 'maybe',
        ]
        lower = stripped.lower()
        for starter in user_starters:
            if lower.startswith(starter.lower()):
                # But only if it's short — long responses starting with these are assistant
                if len(stripped) < 300:
                    return 'user'

    # Strong assistant signals
    for marker in ASSISTANT_MARKERS:
        if stripped.startswith(marker):
            return 'assistant'

    # Long paragraphs are usually assistant
    if len(stripped) > 400:
        return 'assistant'

    # Bullet points and structured content are usually assistant
    if stripped.startswith('- ') or stripped.startswith('• ') or stripped.startswith('| '):
        return 'assistant'

    # Numbered lists
    if re.match(r'^\d+\.?\s', stripped):
        return 'assistant'

    # Code blocks
    if stripped.startswith('```'):
        return 'assistant'

    return None  # keep current role


def _assign_timestamps(messages, year):
    """Assign incremental timestamps based on date headers in the text."""
    for i, msg in enumerate(messages):
        # Parse the base date from the message
        base = msg.get('timestamp', f'{year}-03-18T12:00:00.000Z')
        date_part = base[:10]
        # Add sequential time offset
        hour = 9 + (i * 2) // 60  # spread across the day
        minute = (i * 2) % 60
        if hour > 23:
            hour = 23
            minute = 59
        msg['timestamp'] = f'{date_part}T{hour:02d}:{minute:02d}:00.000Z'


# ── IMPORT TO DB ─────────────────────────────────────────────────────────────

def estimate_tokens(text):
    return max(1, len(text) // 4)


def import_to_db(messages, db_path, project, session_id=None, source_file='', no_embed=False):
    """Import parsed messages into the session memory DB."""
    if not messages:
        print('No messages to import.')
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Ensure schema exists (run migrations from ingest.py)
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

    # Register session
    conn.execute("""
        INSERT OR REPLACE INTO sessions(session_id, source_file, ingested_at, message_count, project)
        VALUES (?, ?, ?, ?, ?)
    """, (session_id, source_file, datetime.now(timezone.utc).isoformat(), len(messages), project))

    inserted = 0
    embedded = 0
    prev_uuid = None

    for i, msg in enumerate(messages):
        chunk_id = str(uuid.uuid4())
        text = msg['text']
        chash = hashlib.sha256(text.encode()).hexdigest()

        # Check for duplicate by content hash
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
            chunk_id,
            session_id,
            prev_uuid,  # linear parent chain
            0,          # not sidechain
            msg['timestamp'],
            msg['role'],
            content_type,
            text,
            estimate_tokens(text),
            chash,
            source_file,
            i,          # seq_index
            project,
        ))

        # FTS trigger handles indexing automatically

        inserted += 1
        prev_uuid = chunk_id

        # Embed if available
        if not no_embed and HAS_EMBED and text and len(text) > 20:
            try:
                vec = embed.get_embedding(text)
                blob = embed.floats_to_blob(vec)
                conn.execute("""
                    INSERT OR IGNORE INTO chunk_embeddings(chunk_id, model, embedding, created_at)
                    VALUES (?, ?, ?, ?)
                """, (chunk_id, embed.EMBED_MODEL, blob, datetime.now(timezone.utc).isoformat()))
                embedded += 1
            except Exception as e:
                pass  # embedding failures are non-fatal

        if (i + 1) % 20 == 0:
            conn.commit()
            print(f'  Progress: {i+1}/{len(messages)} ({inserted} inserted, {embedded} embedded)')

    conn.commit()
    conn.close()

    print(f'  Session  : {session_id[:8]}')
    print(f'  Project  : {project}')
    print(f'  Messages : {len(messages)}')
    print(f'  Inserted : {inserted}')
    if not no_embed:
        print(f'  Embedded : {embedded}')
    print(f'  Done     : {db_path}')


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Import Claude.ai chat conversations into session memory'
    )
    parser.add_argument('file', help='Path to conversation markdown/text file')
    parser.add_argument('--project', required=True, help='Project tag for scoped retrieval')
    parser.add_argument('--db', default=DEFAULT_DB, help='Path to SQLite DB')
    parser.add_argument('--session-date', default=None,
                        help='Default session date (YYYY-MM-DD) if not detected from content')
    parser.add_argument('--year', type=int, default=2026,
                        help='Year for date headers that omit year (default: 2026)')
    parser.add_argument('--session-id', default=None,
                        help='Override session ID (default: auto-generated UUID)')
    parser.add_argument('--no-embed', action='store_true',
                        help='Skip embedding (FTS-only import)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Parse and show results without importing')
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f'ERROR: File not found: {args.file}', file=sys.stderr)
        sys.exit(1)

    print(f'  Source : {args.file}')
    print(f'  Project: {args.project}')

    with open(args.file, encoding='utf-8') as f:
        text = f.read()

    messages = parse_conversation(text, session_date=args.session_date, year=args.year)

    if not messages:
        print('No messages parsed from file.')
        sys.exit(0)

    print(f'  Parsed : {len(messages)} messages '
          f'({sum(1 for m in messages if m["role"]=="user")} user, '
          f'{sum(1 for m in messages if m["role"]=="assistant")} assistant)')

    if args.dry_run:
        print('\n--- DRY RUN ---')
        for i, msg in enumerate(messages):
            role = msg['role'].upper()
            text_preview = msg['text'][:120].replace('\n', ' ')
            tools = ' [tools]' if msg.get('has_tool_use') else ''
            print(f'  [{i:3d}] {role:9s} {msg["timestamp"][:10]}{tools} | {text_preview}')
        return

    import_to_db(
        messages=messages,
        db_path=args.db,
        project=args.project,
        session_id=args.session_id,
        source_file=args.file,
        no_embed=args.no_embed,
    )


if __name__ == '__main__':
    main()
