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
