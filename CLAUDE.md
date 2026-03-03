NEVER run `git checkout -- <files>`, `git checkout .`, `git checkout HEAD`, `git restore .`, `git reset --hard`, `git clean -f`, or ANY command that discards uncommitted changes. NO EXCEPTIONS. Multiple sessions may be running in parallel. If Codex or any tool modifies unexpected files, TELL the user which files and ASK what to do — do NOT revert them.

# edgar-parser — Synced Package Repo

**DO NOT edit code in this repo directly.**

This repo is a deployment artifact synced from the source of truth at `edgar_updater/edgar_parser/`. All code changes must be made there first, then synced here using `edgar_updater/sync_packages.sh`.

See the deploy checklist in the source repo: `edgar_updater/docs/DEPLOY_CHECKLIST.md`