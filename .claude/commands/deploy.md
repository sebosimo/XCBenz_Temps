Commit and push local changes to GitHub, following this project's workflow.

Steps:
1. Run `git status` and `git diff` in parallel with `git log --oneline -5` to understand what changed and the commit style
2. Stage all modified tracked files (ask the user if anything is ambiguous — do NOT stage `.claude/` or secrets)
3. Write a concise commit message (1-2 sentences, imperative mood, explain the "why")
4. Commit using a heredoc, co-authored by Claude
5. Run `git pull --rebase origin main` — CI pushes every 30 min so this is always needed
6. Push to `origin main`
7. Confirm with the commit hash

Important rules for this project:
- Never `git push --force` to main
- Never use `--no-verify`
- Always rebase (not merge) when pulling
- Do not stage `cache_data/`, `cache_wind/`, or any `.nc` files — those are managed by CI only
