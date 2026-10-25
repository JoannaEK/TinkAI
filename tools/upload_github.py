# -*- coding: utf-8 -*-
"""Выгрузка файлов проекта в GitHub-репозиторий через Contents API.

Токен берётся из переменной окружения GITHUB_TOKEN (в файлах не хранится).
Репозиторий задаётся переменными GITHUB_OWNER / GITHUB_REPO / GITHUB_BRANCH,
по умолчанию — JoannaEK/RecSyS, ветка main.
Пути сохраняются относительно корня проекта.
"""
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

OWNER = os.environ.get("GITHUB_OWNER", "JoannaEK")
REPO = os.environ.get("GITHUB_REPO", "RecSyS")
BRANCH = os.environ.get("GITHUB_BRANCH", "main")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/"


def _request(method, url, payload=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url, method=method, data=data,
        headers={
            "Authorization": "Bearer " + os.environ["GITHUB_TOKEN"],
            "User-Agent": "RecSyS-uploader",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        })
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            body = r.read().decode("utf-8")
            return r.status, (json.loads(body) if body else {})
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        return e.code, {"error": body}


def get_sha(remote):
    code, data = _request("GET", API + remote + f"?ref={BRANCH}")
    if code == 200:
        return data.get("sha")
    return None


def upload(local, remote=None):
    remote = remote or os.path.relpath(local, ROOT).replace(os.sep, "/")
    with open(local, "rb") as fh:
        content = base64.b64encode(fh.read()).decode("ascii")
    payload = {"message": f"Add {remote}", "content": content, "branch": BRANCH}
    sha = get_sha(remote)
    if sha:
        payload["sha"] = sha
        payload["message"] = f"Update {remote}"
    code, data = _request("PUT", API + remote, payload)
    if code in (404, 422) and "branch" in str(data).lower():
        # пустой репозиторий: ветки ещё нет, создаём её первым коммитом
        payload.pop("branch", None)
        code, data = _request("PUT", API + remote, payload)
    ok = code in (200, 201)
    print(f"{'OK ' if ok else 'ERR'} {code} {remote}"
          + ("" if ok else " :: " + str(data)[:300]))
    return ok


def collect(paths):
    out = []
    skip_dirs = {"__pycache__", ".git", ".ipynb_checkpoints"}
    for p in paths:
        p = os.path.join(ROOT, p)
        if os.path.isdir(p):
            for dirpath, dirs, files in os.walk(p):
                dirs[:] = [d for d in dirs if d not in skip_dirs]
                for f in files:
                    if f.endswith((".pyc", ".pyo")):
                        continue
                    if f.startswith("upload") and f.endswith(".log"):
                        continue
                    out.append(os.path.join(dirpath, f))
        elif os.path.isfile(p):
            out.append(p)
        else:
            print("missing:", p)
    return sorted(set(out))


def main():
    paths = sys.argv[1:] or [
        "README.md", "requirements.txt",
        "src", "tools", "results", "models", "data/online_retail_II_edited.zip",
    ]
    files = collect(paths)
    print(f"{len(files)} files")
    bad = 0
    for f in files:
        if not upload(f):
            bad += 1
        time.sleep(0.2)
    print(f"done, errors={bad}")


if __name__ == "__main__":
    main()
