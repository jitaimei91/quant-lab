# Frozen Competitors

Snapshots of rival bots that compete in the tournament on identical infrastructure.

## codex-bot-snapshot-2026-05-07

Frozen copy of `morning-quant-bot` (Codex-built) as of 2026-05-07.
**Do not update.** Future updates to the live `morning-quant-bot` repo are intentionally ignored — this is the fixed version against which our bot competes.

To update the snapshot for a future tournament season, run:

```bash
rm -rf competitors/codex-bot-snapshot-<old-date>
cp -R ../morning-quant-bot competitors/codex-bot-snapshot-<new-date>
# clean: .git, .github, __pycache__, state, data
```

The adapter at `src/strategies/codex_bot.py` imports from this frozen snapshot path, NOT the live sibling directory.
