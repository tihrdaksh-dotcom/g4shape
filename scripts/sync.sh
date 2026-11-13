#!/usr/bin/env bash
# Keep this laptop and the GPU workstation in sync (run from anywhere inside the repo).
#
#   scripts/sync.sh push     commit-level code -> GitHub AND the workstation (one `git push`)
#   scripts/sync.sh pull     workstation -> here: runs/ (rsync) + any commits made there
#   scripts/sync.sh status   last commit on each side, and what is not yet pushed
#
# Setup (done once): origin has two push URLs (GitHub + workstation:g4shape), a `workstation`
# remote exists for fetching, and ~/g4shape on the workstation accepts pushes
# (receive.denyCurrentBranch = updateInstead), so its files update on every push.
# Only commits travel by git: commit before `push`. data/ is never copied (it is linked
# to ~/G4_GNN_fast/data on both machines). Nothing is ever deleted on either side.
set -euo pipefail

HOST="${G4_WORKSTATION:-workstation}"     # the Host entry in ~/.ssh/config
REMOTE_DIR="${G4_REMOTE_DIR:-g4shape}"     # relative to the workstation's home directory
cd "$(git rev-parse --show-toplevel)"

case "${1:-}" in
  push)
    if [ -n "$(git status --porcelain)" ]; then
      echo "Note: uncommitted changes stay here (only commits are pushed):"
      git status --short
    fi
    git push origin main --tags
    ;;
  pull)
    echo "== commits made on the workstation"
    git pull --ff-only workstation main
    echo "== runs/ from the workstation"
    if ssh "$HOST" test -d "$REMOTE_DIR/runs"; then
      mkdir -p runs
      rsync -a --human-readable --info=stats1 "$HOST:$REMOTE_DIR/runs/" runs/
    else
      echo "(no runs/ on the workstation yet)"
    fi
    ;;
  status)
    echo "here:        $(git log --oneline -1)"
    echo "workstation: $(ssh "$HOST" "git -C $REMOTE_DIR log --oneline -1")"
    git fetch -q workstation
    echo "commits not on the workstation yet: $(git rev-list --count workstation/main..main)"
    ;;
  *)
    sed -n '2,7p' "$0"
    exit 1
    ;;
esac
