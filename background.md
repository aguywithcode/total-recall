# The Long Story

How and why Total Recall was built, the collaboration process, and the lessons learned.

---

It started with "can I export this conversation?"

I'd just finished a massive Claude Code session, probably 4 hours of architecture decisions, debugging, and design work on an AI runtime I'm building. Hundreds of messages. And I realized all of that context was about to vanish the next time I started a new session.

CLAUDE.md helps, but it's a static summary. I didn't want bullet points. I wanted Claude to actually remember what we talked about, search it, and use it as context going forward.

So I asked Claude to help me build that.

## What I built

A session memory system in ~400 lines of Python that:

- Ingests Claude Code's JSONL transcripts into SQLite
- Embeds every message with a local model (Ollama + nomic-embed-text)
- Indexes for keyword search (FTS5) and vector search (sqlite-vec)
- Walks Claude's conversation DAG (parent message chains, tool call branches) to find surrounding context
- Computes semantic links across sessions and projects

The whole thing is one SQLite file. No server, no API keys, no cloud. Copy it to another machine and it works.

## The DAG thing is important

If you've looked at Claude Code's JSONL transcripts, you'll notice each message has a `parentUuid` field. That's because conversations aren't a flat list. They're a directed acyclic graph. When Claude makes a tool call, the response branches. When you interrupt and redirect, that's a new branch. A single session can have dozens of branches.

This matters for retrieval because a search hit in isolation is often useless. The answer only makes sense if you also have the question that prompted it, and the 3 messages before that where we were narrowing down the problem. The retrieval system walks the parent chain backward from each hit and expands a window around it (configurable, default +/-2 messages). You get the reasoning thread, not a random snippet.

## Project partitioning

I work across a lot of projects: an AI runtime (Arachne), a pitch deck, a book, freelance evaluations, job search, conference planning. They all live in separate Claude Code project directories, which means separate JSONL files.

Each ingested session gets tagged with its project. When Claude queries memory from inside the Arachne project, it only searches Arachne conversations. My universal executive chat searches everything. This keeps results focused. You don't want pitch deck copy showing up when you're debugging a database migration.

## Auto-ingest

A launchd job runs every 15 minutes and ingests new sessions from all my project directories. New chunks get embedded, indexed, and semantically linked automatically. I don't think about it. By the time I start a new session, the knowledge graph already has everything from my last one.

It also pulls in ChatGPT conversations (157 across 30 projects), so the graph spans platforms. A ChatGPT session about agent evaluation design is searchable alongside the Claude Code session where we actually implemented it.

## "Where were we?"

The first thing I type in most new sessions is "where were we?" My CLAUDE.md has instructions to run a `wherewerewe.py` script that shows the last 20 messages from my most recent session (or a specific project). Claude reads those messages and picks up exactly where we left off.

It's the difference between "I'm a new Claude instance, how can I help?" and "Last session you were debugging the WebSocket refactor and got the auth headers working. The asset download was next. Want to pick that up?"

Night and day.

## How Claude uses it

My CLAUDE.md tells every new Claude instance to query memory before answering questions about past work:

```bash
cd ~/Documents/Claude/memory && python3 retrieve.py "your query here"
```

Claude runs this, gets back a `<session_memory>` block with relevant conversation excerpts sorted chronologically, and uses that as context. It takes about a second.

I also turned it into a `/recall` skill so I can just type `/recall arachne deployment architecture` and Claude searches memory, reads the results, and answers with full context from prior sessions.

## The numbers (as of today)

- 27,452 chunks across 47 sessions
- 58,509 semantic links connecting chunks across sessions and projects
- 238 MB single SQLite file
- Auto-ingest every 15 minutes via launchd (all projects, all platforms)
- 157 ChatGPT conversations across 30 projects also ingested

## The part that surprised me

Once you have embeddings for every chunk, you can compute pairwise similarity across your entire conversation history. A discussion about Klarna's AI failure in a YouTube video transcript automatically links to a pattern catalog entry about Semantic Invariants in my book project, which links to an MCP server spec discussion about agent contracts. The graph finds connections I didn't make.

## The ChatGPT import saga

This is the part that shows how working with Claude actually works. It wasn't a straight line. It was four failed approaches before landing on the right one.

**Attempt 1: text dump.** I copied a ChatGPT conversation and pasted it into the session. Claude wrote a Python script with regex patterns and heuristics to detect where user messages ended and assistant messages began: timestamp patterns, tool usage indicators, Q&A formatting, sentence starters that sounded "assistant-like." Clever code. Also wrong. It merged messages that should have been separate, misattributed roles, and fell apart on anything that wasn't a simple back-and-forth.

Claude's instinct was to iterate on the heuristics. Add more patterns, handle more edge cases.

I stopped it: "You parse the conversations. Don't write a script to do it. You read the raw text, you identify each turn, you output structured JSON. I'll import the JSON."

That was the key insight. Claude is better at understanding conversation structure than any regex could be. It can read a messy block of text and understand "this is the user asking a question, this is tool output, this is the assistant correcting itself." That's a comprehension task, not a pattern matching task. Claude parsed the conversations perfectly. One conversation at a time, pasted in, structured JSON out.

**Attempt 2: screenshots.** Pasting worked but didn't scale. I had 157 conversations. So I tried screenshots: scroll through a ChatGPT conversation, take screenshots, feed them to Claude for OCR-style parsing. This sort of worked but was painfully manual and lost formatting.

**Attempt 3: Playwright DOM scraper.** Claude built a Playwright scraper that launched a browser, navigated to ChatGPT, and extracted messages from the DOM. It found user messages fine but got empty strings for assistant responses. ChatGPT virtualizes old messages out of the DOM entirely (only renders what's in the viewport). Programmatic scrolling didn't trigger the re-render. Dead end.

**Attempt 4: Playwright + backend API.** When the DOM scraper hit the virtualization wall, Claude mentioned that ChatGPT has an internal backend API at `/backend-api/conversation/{id}` that returns the full conversation tree. I didn't even know that existed. The idea: keep Playwright for authentication only, then call the API from within the page context using `page.evaluate(fetch(...))`. The browser's cookies handle auth, the API returns every message, real timestamps, and model metadata.

But it still took 5 rounds of debugging. The API needed extra headers beyond cookies (authorization bearer token, `oai-device-id`, `oai-client-build-number`). My conversations lived inside ChatGPT Projects, not at the top level, so the conversation list endpoint returned nothing until we discovered the `/gizmos/snorlax/sidebar` endpoint. The JSON was nested one level deeper than expected (`item.gizmo.gizmo.id` instead of `item.gizmo.id`).

The pattern was consistent: Claude writes code based on assumptions about an undocumented API. I run it, report what actually happens (often by reading my browser's DevTools network tab, something Claude can't see). Claude adapts. Neither of us could have done it alone.

One script now discovers all 30 of my ChatGPT projects and scrapes 157 conversations in minutes.

We kept going. The scraper was capturing text but dropping all the images: DALL-E generations, screenshots I'd uploaded, code interpreter outputs. They all showed up as `[image]` placeholders. So we added asset downloading. The scraper now navigates to the conversation page, extracts signed image URLs from the rendered DOM, downloads them through the authenticated session, and saves them alongside the conversation JSON. DALL-E prompts are preserved in the metadata. Uploaded file attachments get their filename, mime type, and size recorded. Code interpreter output captures the executed Python and its results.

The graph doesn't care where a conversation came from, and now it doesn't lose the artifacts either.

## What it looks like in practice

Me: "what did we decide about the deployment architecture?"

Claude runs `retrieve.py "deployment architecture"`, gets back 20 chunks from 3 different sessions spanning 2 weeks, and answers with specifics: which cloud provider, which container strategy, which CI/CD pipeline, and why we chose them.

No re-explaining. No "as a new session I don't have context." Just the answer, grounded in actual prior conversations.

## Stack

- Python 3.13
- SQLite (FTS5 + sqlite-vec)
- Ollama + nomic-embed-text (local embeddings)
- Zero cloud dependencies

## Making it available

I'm packaging this into a repo so others can set up their own persistent memory for Claude. The development team is a Claude Agent Team (the new multi-agent feature that just dropped this week) themed after Matrix characters: Trinity handles UX, Switch does frontend, Tank builds the backend, Mouse writes tests, Agent Smith does code review, and Morpheus runs the sprints. The full crew. They're currently working on the knowledge graph repo to make it installable and documented for other Claude Code users.

The goal is: clone the repo, run the ingest, add a few lines to your CLAUDE.md, and every new session has full recall of everything you've ever discussed.

## What's next

- Web UI for browsing and managing the knowledge graph
- Agent-scoped knowledge bases (each team member agent gets its own RAG pipeline)
- Cross-platform graph visualization
- Making the whole thing a one-command install

Happy to answer questions about the architecture or share more details on specific parts.
