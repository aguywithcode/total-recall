---
name: Universal session model
description: Michael uses one persistent "executive context" chat that knows everything, plus project-specific chats whose transcripts get rolled up into session memory
type: user
---

Michael's working model: this Claude session is his universal/executive chat — the one that maintains full context about him, his companies, decisions, and ongoing work across all projects. He creates separate project-specific chats for focused work, then ingests their JSONL transcripts into the session memory DB so this universal chat stays aware of everything happening across his world.

Implication: when Michael asks about something, always query session memory first — the answer may have come from a project-specific chat that was rolled up.
