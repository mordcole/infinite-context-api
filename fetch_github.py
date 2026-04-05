import os
import json
import hashlib
import requests
from datetime import datetime

REPOS = [
    "ls-4/RequestBin",
    "mordcole/notes-backend",
    "mordcole/phonebook-backend",
    "mordcole/phonebook-frontend",
    "mordcole/notes-frontend",
    "mordcole/static-site-api",
    "mordcole/counter-app",
    "mordcole/book-viewer",
    "mordcole/mordecai-me",
    "mordcole/hello-world",
]

OUTPUT_DIR = os.path.expanduser("~/Launch_School/Capstone/rag/data/github")
GITHUB_API = "https://api.github.com"
HEADERS = {"Accept": "application/vnd.github+json"}
MAX_DIFF_CHARS = 3000


def fetch_commits(repo):
    commits = []
    page = 1
    while True:
        url = f"{GITHUB_API}/repos/{repo}/commits"
        response = requests.get(url, headers=HEADERS, params={"per_page": 100, "page": page})
        if response.status_code != 200:
            print(f"Skipping {repo}: {response.status_code}")
            break
        batch = response.json()
        if not batch:
            break
        commits.extend(batch)
        page += 1
    return commits


def fetch_diff(repo, sha):
    url = f"{GITHUB_API}/repos/{repo}/commits/{sha}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        return ""

    data = response.json()
    diff_lines = []
    for file in data.get("files", []):
        filename = file.get("filename", "")
        patch = file.get("patch", "")
        if patch:
            diff_lines.append(f"--- {filename} ---\n{patch}")

    full_diff = "\n".join(diff_lines)
    return full_diff[:MAX_DIFF_CHARS]


def format_commit(commit, repo, diff):
    sha = commit["sha"][:7]
    message = commit["commit"]["message"].strip()
    author = commit["commit"]["author"]["name"]
    date = commit["commit"]["author"]["date"]

    text = f"Repo: {repo}\nCommit: {sha}\nAuthor: {author}\nDate: {date}\nMessage: {message}"
    if diff:
        text += f"\nDiff:\n{diff}"

    return {
        "id": hashlib.md5(f"{repo}:{commit['sha']}".encode()).hexdigest(),
        "text": text,
        "metadata": {
            "source": "github",
            "repo": repo,
            "sha": sha,
            "author": author,
            "date": date,
            "filename": f"{repo.replace('/', '_')}_{sha}",
        }
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_chunks = []

    for repo in REPOS:
        print(f"Fetching {repo}...")
        commits = fetch_commits(repo)
        print(f"  {len(commits)} commits found, fetching diffs...")
        for i, commit in enumerate(commits):
            diff = fetch_diff(repo, commit["sha"])
            all_chunks.append(format_commit(commit, repo, diff))
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(commits)} commits processed...")

    output_path = os.path.join(OUTPUT_DIR, "commits.json")
    with open(output_path, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"\nDone. {len(all_chunks)} commits written to {output_path}")


if __name__ == "__main__":
    main()
