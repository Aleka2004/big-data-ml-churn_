#!/usr/bin/env bash
set -euo pipefail

REPO_NAME="big-data-ml-churn"
OWNER="Aleka2004"

if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI (gh) орнатылмаған. Орнату: https://cli.github.com/"
  exit 1
fi

gh auth status || gh auth login

git init
git add .
git commit -m "Initial Big Data ML project" || true
git branch -M main

gh repo create "${OWNER}/${REPO_NAME}" --public --source=. --remote=origin --push

echo "Repository: https://github.com/${OWNER}/${REPO_NAME}"
