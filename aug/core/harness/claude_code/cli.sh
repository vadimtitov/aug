#!/usr/bin/env bash
# Starts the Claude Code CLI with only the environment variables named in
# AUG_CLAUDE_ENV_KEEP, so AUG's own secrets (API keys, DATABASE_URL, ...) never reach it.
# The SDK always passes the parent's whole environment; this is where it is cut.
keep=()
for name in $AUG_CLAUDE_ENV_KEEP; do
  if [ -n "${!name+x}" ]; then
    keep+=("$name=${!name}")
  fi
done
exec env -i "${keep[@]}" "$AUG_CLAUDE_CLI" "$@"
