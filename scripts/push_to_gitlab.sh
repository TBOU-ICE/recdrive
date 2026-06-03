#!/usr/bin/env bash
# Push main to GitLab. Requires GITLAB_TOKEN (Personal Access Token with write_repository).
# Usage: GITLAB_TOKEN='glpat-...' bash scripts/push_to_gitlab.sh
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

REMOTE_URL="${REMOTE_URL:-https://gitlab.hellorobotaxi.top/niebingyan612/recdrive-opd.git}"
BRANCH="${BRANCH:-main}"

if [ -z "${GITLAB_TOKEN:-}" ]; then
  echo "error: set GITLAB_TOKEN (GitLab PAT with write_repository scope)" >&2
  exit 1
fi

# GitLab HTTPS through the local proxy often breaks TLS; push direct.
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
  git push "https://oauth2:${GITLAB_TOKEN}@${REMOTE_URL#https://}" "${BRANCH}"
