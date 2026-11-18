#!/usr/bin/env bash
# Install AUG on this machine:
#
#   curl -fsSL https://raw.githubusercontent.com/vadimtitov/aug/main/install.sh | bash
#
# Clones the repository into $AUG_DIR (default ~/aug) and runs ./augctl setup, which
# asks for a Telegram bot token, writes .env and starts AUG in Docker.  Running it
# again updates an existing checkout and keeps its .env.
set -euo pipefail

repo=${AUG_REPO:-https://github.com/vadimtitov/aug.git}
dir=${AUG_DIR:-$HOME/aug}

for tool in git docker curl; do
  command -v "$tool" > /dev/null || { echo "error: $tool is not installed." >&2; exit 1; }
done

if [ -d "$dir/.git" ]; then
  echo "Updating $dir"
  git -C "$dir" pull --ff-only || {
    echo "error: git could not update $dir (above). Local changes? git stash, then retry." >&2
    exit 1
  }
elif [ -e "$dir" ]; then
  echo "error: $dir exists and is not an AUG checkout. Set AUG_DIR to install elsewhere." >&2
  exit 1
else
  echo "Cloning AUG into $dir"
  git clone --depth 1 "$repo" "$dir"
fi

exec "$dir/augctl" setup
