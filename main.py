import json
import time
import urllib.request

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Union

from synthesize import synthesize, rewrite_query
from retrieve import retrieve
from embed import clean_text, split_text, hash_text, generate_embeddings, upsert_chunks

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

SCORE_THRESHOLD = 0.20


class FileItem(BaseModel):
    filename: str
    content: str


class IngestNotesRequest(BaseModel):
    files: List[FileItem]
    source: Optional[str] = "remnote"


class IngestNotesResponse(BaseModel):
    ingested: int
    chunks: int


class IngestGitHubRequest(BaseModel):
    pat: str
    repo: str  # "owner/reponame"


class IngestGitHubResponse(BaseModel):
    ingested: int
    chunks: int


class ChatRequest(BaseModel):
    query: str
    history: List[Dict[str, str]] = []


class SourceItem(BaseModel):
    filename: str
    type: str
    score: float
    text: str


class ChatResponse(BaseModel):
    answer: str
    history: List[Dict[str, str]]
    sources: List[SourceItem]
    mode: str
    coverage: Union[str, None] = None

def clean_filename(filename):
    name = filename.split("::")[0]
    name = name.replace(".md", "")
    return name

@app.post("/ingest/notes", response_model=IngestNotesResponse)
def ingest_notes(request: IngestNotesRequest):
    try:
        chunks = []
        for file in request.files:
            text = clean_text(file.content)
            if not text:
                continue
            splits = split_text(text) if len(text) > 6000 else [text]
            for i, split in enumerate(splits):
                chunks.append({
                    "text": split,
                    "source": request.source,
                    "date": "",
                    "doc_id": hash_text(split),
                    "filename": f"{file.filename}::chunk{i}"
                })

        if chunks:
            chunks = generate_embeddings(chunks)
            upsert_chunks(chunks)

        return IngestNotesResponse(ingested=len(request.files), chunks=len(chunks))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/github", response_model=IngestGitHubResponse)
def ingest_github(request: IngestGitHubRequest):
    try:
        owner, reponame = request.repo.split("/", 1)
        gh_headers = {
            "Authorization": f"token {request.pat}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "infinite-context-rag",
        }

        # Fetch all commits with pagination
        all_commits = []
        page = 1
        while True:
            url = f"https://api.github.com/repos/{request.repo}/commits?per_page=100&page={page}"
            req = urllib.request.Request(url, headers=gh_headers)
            with urllib.request.urlopen(req) as resp:
                page_data = json.loads(resp.read())
            if not page_data:
                break
            all_commits.extend(page_data)
            if len(page_data) < 100:
                break
            page += 1
            time.sleep(0.5)

        # For each commit, fetch full detail and build chunks
        chunks = []
        filename_prefix = f"{owner}_{reponame}"

        for commit in all_commits:
            sha = commit["sha"]
            url = f"https://api.github.com/repos/{request.repo}/commits/{sha}"
            req = urllib.request.Request(url, headers=gh_headers)
            with urllib.request.urlopen(req) as resp:
                detail = json.loads(resp.read())

            message = detail["commit"]["message"]
            author = detail["commit"]["author"]["name"]
            date = detail["commit"]["author"]["date"]
            files = detail.get("files", [])

            parts = [f"commit: {sha[:8]}\nmessage: {message}\nauthor: {author}\ndate: {date}"]
            for f in files:
                patch = f.get("patch", "")
                if patch:
                    parts.append(f"{f['filename']}:\n{patch}")

            text = clean_text("\n---\n".join(parts))
            if not text:
                time.sleep(0.1)
                continue

            splits = split_text(text) if len(text) > 6000 else [text]
            filename = f"{filename_prefix}_{sha}"
            for i, split in enumerate(splits):
                chunks.append({
                    "text": split,
                    "source": "github",
                    "date": date,
                    "doc_id": hash_text(split),
                    "filename": f"{filename}::chunk{i}",
                })

            time.sleep(0.1)  # rate limit buffer

        if chunks:
            chunks = generate_embeddings(chunks)
            upsert_chunks(chunks)

        return IngestGitHubResponse(ingested=len(all_commits), chunks=len(chunks))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        # Run synthesis (includes query rewrite, classification, and coverage assessment)
        updated_history, mode, coverage = synthesize(request.query, request.history)
        answer = updated_history[-1]["content"]

        # Skip retrieval for off-topic queries
        if mode == "off_topic":
            return ChatResponse(answer=answer, history=updated_history, sources=[], mode=mode, coverage=None)

        # Retrieve matches and build source list for learning queries
        rewritten = rewrite_query(request.query, request.history)
        matches = retrieve(rewritten)
        relevant = [m for m in matches if m.score >= SCORE_THRESHOLD]
        sources = [
            SourceItem(
                filename=clean_filename(m.metadata.get("filename", "unknown")),
                type="GITHUB" if m.metadata.get("source") == "github" else "TRANSCRIPTS" if m.metadata.get("source") == "transcripts" else "REMNOTE",
                score=round(m.score, 2),
                text=m.metadata.get("text", ""),
            )
            for m in relevant[:5]
        ]

        return ChatResponse(answer=answer, history=updated_history, sources=sources, mode=mode, coverage=coverage)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
