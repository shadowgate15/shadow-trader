## Tooling

Python 3.13 project managed with [uv](https://docs.astral.sh/uv/). Always run Python tooling through `uv run` rather than invoking tools directly.

- **Dependencies / environment:** `uv sync` to install, `uv add <pkg>` (or `uv add --dev <pkg>`) to add packages. Don't use `pip`.
- **Linting and formatting:** [ruff](https://docs.astral.sh/ruff/) — `uv run ruff check` and `uv run ruff format`. Config is in `pyproject.toml`.
- **Type checking:** [ty](https://docs.astral.sh/ty/) — `uv run ty check`. Config is in `pyproject.toml`.

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues (shadowgate15/shadow-trader), via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
