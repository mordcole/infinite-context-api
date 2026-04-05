import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("infinite-context")


def embed_query(query):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    return response.data[0].embedding


def retrieve(query, top_k=9):
    vector = embed_query(query)
    per_source = top_k // 3

    remnote_results = index.query(
        vector=vector,
        top_k=per_source,
        include_metadata=True,
        filter={"source": "remnote"}
    )
    github_results = index.query(
        vector=vector,
        top_k=per_source,
        include_metadata=True,
        filter={"source": "github"}
    )
    transcript_results = index.query(
        vector=vector,
        top_k=per_source,
        include_metadata=True,
        filter={"source": "transcripts"}
    )

    all_matches = remnote_results.matches + github_results.matches + transcript_results.matches
    all_matches.sort(key=lambda x: x.score, reverse=True)
    return all_matches


if __name__ == "__main__":
    query = input("Query: ").strip()
    matches = retrieve(query)
    for i, match in enumerate(matches):
        print(f"--- Result {i + 1} ---")
        print(f"Score:    {match.score:.4f}")
        print(f"Source:   {match.metadata['source']}")
        print(f"Filename: {match.metadata['filename']}")
        print(f"Text:     {match.metadata['text'][:300]}")
        print()
