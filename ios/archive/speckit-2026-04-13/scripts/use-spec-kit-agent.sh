#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/use-spec-kit-agent.sh <claude|codex>

Switch the active Spec Kit integration for this repo while preserving
the existing agent skill directories on disk.

Examples:
  scripts/use-spec-kit-agent.sh claude
  scripts/use-spec-kit-agent.sh codex
EOF
}

if [[ $# -ne 1 ]]; then
  usage
  exit 1
fi

agent="$1"

case "$agent" in
  claude|codex)
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "Unsupported agent: $agent" >&2
    usage
    exit 1
    ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$repo_root"

if [[ ! -d .specify ]]; then
  echo "Spec Kit is not initialized in $repo_root" >&2
  exit 1
fi

specify init --here --ai "$agent" --ai-skills --force --ignore-agent-tools

echo
echo "Active Spec Kit integration switched to '$agent'."
echo "Current integration metadata:"
cat .specify/integration.json
