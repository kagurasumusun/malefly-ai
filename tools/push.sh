#!/bin/sh
# re-establishes remote/user each run (.git/config is not persisted in sandbox)
set -e
cd "$(dirname "$0")/.."
TOKEN="${MALEFLY_TOKEN:-$(cat tools/token.local 2>/dev/null)}"
[ -n "$TOKEN" ] || { echo "no token: put PAT in tools/token.local" >&2; exit 1; }
git config user.name kagurasumusun
git config user.email kagurasumusun@users.noreply.github.com
if ! git remote | grep -q '^origin$'; then
  git remote add origin "https://x-access-token:${TOKEN}@github.com/kagurasumusun/malefly-ai.git"
fi
exec git "$@"
