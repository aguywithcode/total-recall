# Total Recall

Persistent memory for Claude Code. Every new session has full recall of everything you've ever discussed.

One SQLite file. No server, no API keys, no cloud. Copy it to another machine and it works.

---

## The Problem

Every new Claude Code session starts with amnesia. CLAUDE.md helps, but it's a static summary. Total Recall gives Claude searchable access to your actual conversation history: every decision, every debugging session, every architecture discussion.

## What It Does

- Ingests Claude Code's JSONL transcripts into SQLite
- Embeds every message with a local model (Ollama + nomic-embed-text)
- Indexes for keyword search (FTS5) and vector search (sqlite-vec)
- Walks Claude's conversation DAG (parent message chains, tool call branches) to find surrounding context
- Computes semantic links across sessions and projects
- Also scrapes ChatGPT conversations via backend API (with asset/image downloads)

## The DAG Thing Is Important

Claude Code's JSONL transcripts aren't flat lists. Each message has a `parentUuid` field forming a directed acyclic graph. When Claude makes a tool call, the response branches. When you interrupt and redirect, that's a new branch.

This matters for retrieval because a search hit in isolation is often useless. The answer only makes sense with the question that prompted it. The retrieval system walks the parent chain backward from each hit and expands a window around it (configurable, default +/-2 messages). You get the reasoning thread, not a random snippet.

## Project Partitioning

Each ingested session gets tagged with its project. Queries from inside a project only search that project's conversations. A universal chat searches everything. You don't want pitch deck copy showing up when you're debugging a database migration.

## "Where Were We?"

The first thing you type in a new session: "where were we?" The `wherewerewe.py` script shows the last 20 messages from your most recent session. Claude reads them and picks up exactly where you left off.

The difference between "I'm a new Claude instance, how can I help?" and "Last session you were debugging the WebSocket refactor. The asset download was next. Want to pick that up?"

---

## Quick Start

### 1. Prerequisites

```bash
# Install Ollama (https://ollama.ai)
brew install ollama    # macOS
ollama pull nomic-embed-text

# Python dependencies
pip install requests apsw
```

### 2. Clone

```bash
git clone https://github.com/aguywithcode/total-recall.git
cd total-recall
```

### 3. Ingest Your Sessions

```bash
# Ingest a single session
python3 ingest.py ~/.claude/projects/YOUR_PROJECT/*.jsonl --db memory.db

# Ingest all sessions from a project
for f in ~/.claude/projects/YOUR_PROJECT/*.jsonl; do
  python3 ingest.py "$f" --db memory.db --no-embed
done
```

### 4. Embed

```bash
# Make sure Ollama is running
ollama serve &

# Embed all chunks
python3 backfill_embeddings.py --db memory.db
```

### 5. Build Semantic Links

```bash
python3 semantic_linker.py --db memory.db
```

### 6. Search

```bash
python3 retrieve.py "your query here" --db memory.db
```

### 7. Resume a Session

```bash
# Last 20 messages from most recent session
python3 wherewerewe.py --db memory.db

# Last 20 from a specific project
python3 wherewerewe.py --db memory.db --project myproject

# More context
python3 wherewerewe.py --db memory.db -n 50
```

---

## Integrate with Claude Code

Add this to your project's `CLAUDE.md`:

````markdown
## Session Memory

Before answering questions about past decisions, query session memory:

```bash
cd /path/to/total-recall && python3 retrieve.py "your query here" --db memory.db
```

To resume a previous session:

```bash
python3 wherewerewe.py --db memory.db
```
````

### Set Up the /recall Skill

Create `~/.claude/skills/recall/SKILL.md`:

````markdown
---
name: recall
description: Search session memory for prior conversations, decisions, and context
user-invocable: true
---

# Recall

Run the retrieval script:

```bash
cd /path/to/total-recall && python3 retrieve.py "{{query}}" --db memory.db
```

Read the output and summarize what you found.
````

Then type `/recall deployment architecture` in any session.

### Auto-Permission (No Prompts)

Add to your `~/.claude/settings.json`:

```json
{
  "permissions": {
    "allow": [
      "Bash(python3 /path/to/total-recall/retrieve.py *)"
    ]
  }
}
```

---

## Auto-Ingest (Optional)

Set up a cron job or launchd plist to ingest new sessions automatically:

```bash
# Example: run every 15 minutes
*/15 * * * * cd /path/to/total-recall && for f in ~/.claude/projects/*/*.jsonl; do python3 ingest.py "$f" --db memory.db --no-embed 2>/dev/null; done && python3 backfill_embeddings.py --db memory.db 2>/dev/null && python3 semantic_linker.py --db memory.db 2>/dev/null
```

---

## Retrieval Options

```bash
# Wider context window (more surrounding messages)
python3 retrieve.py "your query" --window 3 --budget 25

# More initial matches, deeper ancestor traversal
python3 retrieve.py "your query" --top-k 8 --depth 4

# Only user messages
python3 retrieve.py "your query" --roles user

# Skip tool call noise
python3 retrieve.py "your query" --no-tools

# JSON output for programmatic use
python3 retrieve.py "your query" --format json
```

---

## ChatGPT Scraper

Import conversations from ChatGPT (uses Playwright for auth, then calls ChatGPT's backend API):

```bash
# Install Playwright
pip install playwright
playwright install chromium

# List all ChatGPT projects
python3 scrape_chatgpt.py --list-projects

# Scrape a single conversation
python3 scrape_chatgpt.py --url "https://chatgpt.com/c/abc123" --project myproject --stage

# Scrape all conversations in a ChatGPT project
python3 scrape_chatgpt.py --chatgpt-project "My Project" --project myproject --stage

# Without downloading images
python3 scrape_chatgpt.py --chatgpt-project "My Project" --project myproject --stage --no-assets
```

First run opens a browser for manual login. Subsequent runs reuse the saved session.

---

## Architecture

**Three search dimensions:**

- **Keyword (FTS5):** Porter-stemmed full-text search. Fast, good for exact terms.
- **Semantic (sqlite-vec KNN):** Vector similarity over embeddings. Finds conceptually related content.
- **RRF Fusion:** Reciprocal Rank Fusion merges both ranked lists.

**Three context expansion strategies:**

- **Window expansion:** Include surrounding messages around each hit (+/-2 by default).
- **Ancestor backtracking:** Walk the parentUuid DAG backward to find the reasoning thread.
- **Semantic link traversal:** Cross-session connections via the `semantic_links` table.

## Stack

- Python 3.13
- SQLite (FTS5 for keyword search)
- sqlite-vec (KNN vector search)
- Ollama + nomic-embed-text (local embeddings, 768-dim)
- apsw (SQLite wrapper with extension support)
- Playwright (ChatGPT scraper only)
- Zero cloud dependencies

Everything runs locally. The embeddings are generated by Ollama on your machine. The entire knowledge graph lives in a file you control.

---

## What's Next

- Web UI for browsing and managing the knowledge graph
- Agent-scoped knowledge bases
- Cross-platform graph visualization
- One-command install

---

## License

MIT

---

Built by [Michael Brown](https://github.com/aguywithcode) with Claude Code.
