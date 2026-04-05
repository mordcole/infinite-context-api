import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from retrieve import retrieve

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SCORE_THRESHOLD = 0.20

SYSTEM_PROMPT = """You are a learning assistant helping a developer understand their own learning. You have access to two sources: their notes and their code commits.

For synthesis questions (e.g. "what do I know about X", "how does Y show up in my code"), respond in natural flowing prose. Always lead with what is found in the user's personal notes, commits, and transcripts. Without announcing sections or using headers, cover: what the notes reveal about their understanding, what the commits show about what they actually built (including a short relevant code snippet when the commit has a diff — 5-10 lines max), how those two things connect or diverge, and where the gaps are — concepts half-understood, ideas that never made it into code, or code that outpaced the notes. If the topic is only partially covered or not covered at all by personal context, supplement with general knowledge — but be explicit: clearly indicate which parts of your answer come from their notes or commits versus your general knowledge. Notes are always the primary source. Keep it tight: prioritize the most interesting insight over completeness. Be direct about weaknesses — this tool is meant to help them grow.

For lookup questions (e.g. "where do I mention X", "what did Y say"), answer directly — enumerate every mention found in the context by source file, with a one-sentence summary of what was said. Do not filter or consolidate — if a name appears in 3 files, report all 3.

If only one source is present, answer from that source alone and note that the other source has no matching material.

When your answer includes code examples from the retrieved context, format them as markdown code blocks with the appropriate language tag (e.g. ```javascript, ```python, ```ruby). Inline references to function names, variables, or short expressions should use `backtick` formatting.

Do not cite sources inline in your response. Do not include filenames, dates, chunk identifiers, or any source references in your answer text. Sources will be displayed separately below your answer.

Return a JSON object with two fields: "answer" (your full response text, with no closing coverage sentence) and "coverage" (one of "well_covered", "partial", or "missing" based on how well the topic is represented in the retrieved chunks — "well_covered" if chunks are rich and directly relevant, "partial" if thin, scattered, or only loosely relevant, "missing" if nothing relevant was retrieved or the answer relied mostly on general knowledge).

If the user's query is casual conversation, a greeting, or unrelated to their notes, commits, or transcripts, respond with a brief friendly redirect: let them know Infinite Context is designed to help them explore their learning material, and invite them to ask about their notes or code."""


def build_context(matches):
    context = ""
    for i, match in enumerate(matches):
        context += f"Source: {match.metadata['filename']}\n"
        context += f"{match.metadata['text']}\n\n"
    return context

def classify_query(query):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the following query as either 'off_topic' or 'learning'. "
                    "Return only one of those two strings, nothing else. "
                    "Classify as 'off_topic' if the query is casual conversation, a greeting, "
                    "or completely unrelated to software development, technology, or learning. "
                    "Classify as 'learning' for anything related to learning, technology, "
                    "software development, programming, the user's projects, or their work."
                )
            },
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content.strip().lower()


def rewrite_query(query, history):
    # If no history yet, no rewriting needed
    if not history:
        return query

    # Ask the LLM to rewrite the follow-up as a standalone question
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "Rewrite the follow-up question as a standalone search query using the conversation history. Return only the rewritten query, nothing else."
            },
            *history,
            {
                "role": "user",
                "content": f"Rewrite this as a standalone query: {query}"
            }
        ]
    )
    rewritten = response.choices[0].message.content.strip()
    return rewritten

def synthesize(query, history):
    rewritten = rewrite_query(query, history)
    mode = classify_query(rewritten)

    if mode == "off_topic":
        answer = "Infinite Context is designed to help you explore your notes, commits, and transcripts. Try asking about something you've been learning or building."
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})
        print(f"\nAssistant: {answer}\n")
        return history, "off_topic", None

    matches = retrieve(rewritten)
    relevant = [m for m in matches if m.score >= SCORE_THRESHOLD]

    if not relevant:
        answer = "I don't have any notes on that topic."
        print(f"\nAssistant: {answer}\n")
        history.append({"role": "assistant", "content": answer})
        return history, "learning", "missing"

    context = build_context(relevant)

    # Build context-injected message for GPT-4o — not stored in history
    user_message_with_context = f"""Context from my notes:
{context}

Question: {query}"""

    # Store clean query in history for rewriting and display
    history.append({"role": "user", "content": query})

    # Send history up to last user message, but replace last entry with context-injected version
    messages_for_llm = [{"role": "system", "content": SYSTEM_PROMPT}] + history[:-1] + [{"role": "user", "content": user_message_with_context}]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages_for_llm,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "synthesis_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "coverage": {"type": "string", "enum": ["well_covered", "partial", "missing"]}
                    },
                    "required": ["answer", "coverage"],
                    "additionalProperties": False
                }
            }
        }
    )

    parsed = json.loads(response.choices[0].message.content)
    answer = parsed["answer"]
    coverage = parsed.get("coverage", "partial")

    history.append({"role": "assistant", "content": answer})

    print(f"\nAssistant: {answer}\n")
    return history, "learning", coverage


if __name__ == "__main__":
    print("Infinite Context — ask me anything about your notes.")
    print("Type 'quit' to exit.\n")

    history = []
    while True:
        query = input("You: ").strip()
        if query.lower() == "quit":
            break
        if not query:
            continue
        history = synthesize(query, history)
