import os
import re
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# Load API keys from .env
load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("infinite-context")

REMNOTE_DIR = Path("data/remnote")
GITHUB_DIR = Path("data/github")
TRANSCRIPTS_DIR = Path("data/transcripts")


def clean_text(text):
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'https?://\S+', '', text)
    return text.strip()


def hash_text(text):
    return hashlib.md5(text.encode()).hexdigest()


def split_text(text, chunk_size=6000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


def load_remnote_chunks():
    chunks = []
    for path in Path(REMNOTE_DIR).rglob("*.md"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        text = clean_text(text)
        if not text:
            continue

        splits = split_text(text) if len(text) > 6000 else [text]

        for i, split in enumerate(splits):
            chunks.append({
                "text": split,
                "source": "remnote",
                "date": "",
                "doc_id": hash_text(split),
                "filename": f"{path.name}::chunk{i}"
            })
    return chunks


def load_transcript_chunks():
    chunks = []
    for path in Path(TRANSCRIPTS_DIR).rglob("*.md"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        text = clean_text(text)
        if not text:
            continue
        splits = split_text(text) if len(text) > 6000 else [text]
        for i, split in enumerate(splits):
            chunks.append({
                "text": split,
                "source": "transcripts",
                "date": "",
                "doc_id": hash_text(split),
                "filename": f"{path.name}::chunk{i}"
            })
    return chunks


def load_github_chunks():
    commits_path = GITHUB_DIR / "commits.json"
    if not commits_path.exists():
        print("No commits.json found, skipping GitHub.")
        return []

    with open(commits_path) as f:
        commits = json.load(f)

    chunks = []
    for commit in commits:
        chunks.append({
            "text": commit["text"],
            "source": "github",
            "date": commit["metadata"]["date"],
            "doc_id": commit["id"],
            "filename": commit["metadata"]["filename"]
        })
    return chunks


def generate_embeddings(chunks, batch_size=25):
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [chunk["text"] for chunk in batch]
        print(f"  Embedding batch {i // batch_size + 1} ({len(batch)} chunks)...")
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        for j, item in enumerate(response.data):
            batch[j]["embedding"] = item.embedding
    return chunks


def upsert_chunks(chunks, batch_size=100):
    vectors = [{
        "id": chunk["doc_id"],
        "values": chunk["embedding"],
        "metadata": {
            "text": chunk["text"][:4000],
            "source": chunk["source"],
            "filename": chunk["filename"],
            "date": chunk["date"]
        }
    } for chunk in chunks]

    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f"  Upserted batch {i // batch_size + 1} ({len(batch)} vectors)...")

    print(f"Upserted {len(vectors)} vectors total")


if __name__ == "__main__":
    import sys
    source = sys.argv[1] if len(sys.argv) > 1 else "all"

    chunks = []

    if source in ("all", "remnote"):
        print("Loading RemNote chunks...")
        remnote_chunks = load_remnote_chunks()
        print(f"Found {len(remnote_chunks)} RemNote chunks")
        chunks += remnote_chunks

    if source in ("all", "github"):
        print("Loading GitHub chunks...")
        github_chunks = load_github_chunks()
        print(f"Found {len(github_chunks)} GitHub chunks")
        chunks += github_chunks

    if source in ("all", "transcripts"):
        print("Loading transcript chunks...")
        transcript_chunks = load_transcript_chunks()
        print(f"Found {len(transcript_chunks)} transcript chunks")
        chunks += transcript_chunks

    print(f"Total: {len(chunks)} chunks")
    print("Generating embeddings...")
    chunks = generate_embeddings(chunks)

    print("Upserting to Pinecone...")
    upsert_chunks(chunks)
    print("Done.")
