# Infinite Context — Backend Claude Code Briefing

Personal RAG application for software engineering students.
Semantic search and AI synthesis across three personal learning
sources: notes, GitHub commits, and pairing transcripts.

## Project Location
~/Launch_School/Capstone/rag/

## Running the Backend
```bash
source .venv/bin/activate
uvicorn main:app --reload
```
Backend runs on localhost:8000.

## Tech Stack
- Python, FastAPI, uvicorn
- OpenAI (text-embedding-3-small for embeddings, GPT-4o for synthesis)
- Pinecone (index: infinite-context, 1536 dims, cosine, AWS us-east-1)
- No LangChain — keep all code simple and direct

## File Structure
rag/
embed.py        — chunking, hashing, embedding, upserting to Pinecone
retrieve.py     — parallel source-filtered queries, top-3 per source
synthesize.py   — query classification, rewriting, GPT-4o synthesis
main.py         — FastAPI routes: POST /chat, POST /ingest/notes
data/
remnote/      — exported .md note files
github/       — commit data
transcripts/  — pairing session .md files
.env            — OPENAI_API_KEY, PINECONE_API_KEY (never touch)
.venv/          — Python venv (never touch)

## Pinecone Metadata Schema
Each vector has: text, source, filename, score
source values: "remnote" | "github" | "transcripts"

## Rules — Read Before Writing Any Code
- Read existing files before editing or extending them
- Follow patterns already established in embed.py and synthesize.py
- Use structured outputs (response_format JSON schema) for any new
  GPT-4o calls — same pattern as synthesize.py
- Never modify .env
- Never modify or re-run embed.py against existing data
  (would cause duplicate vectors)
- Install new pip packages with --break-system-packages flag
- Always activate .venv before running anything
- Surgical edits only — do not refactor working code

## What's Already Built and Working
- POST /chat — full RAG pipeline with query rewriting,
  classification, parallel retrieval, structured synthesis
- POST /ingest/notes — accepts markdown files in batches,
  chunks, embeds, upserts with source param
- ~2,853 vectors live in Pinecone across three sources

## Current Task
See task prompt provided at session start.
