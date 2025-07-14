# Contributing to Puma

First off, **thank you** for taking the time to contribute — every bug‑fix, feature and documentation improvement helps the ATLAS flavour‑tagging community.

If this is your first visit, please start with our detailed [Developer Guidelines](https://umami-hep.github.io/puma/dev/development_guidelines/). They cover environment setup, branching strategy, coding style and more. What follows is a boiled‑down checklist.

---

## Quick start

1. **Fork** the repository and clone it locally.
2. Create and activate a Python ≥ 3.9 virtual environment (or conda env).
3. Install the package *plus* development extras:

   ```bash
   python -m pip install -e ".[dev]"
   pre-commit install
   ```
4. Run the hook chain once to auto‑format and lint the existing code base:

   ```bash
   pre-commit run --all-files
   ```

---

## Making changes

| Step | Command | Notes |
|------|---------|-------|
| **Create a feature branch** | `git checkout -b feature/your-topic` | Keep branches tightly scoped. |
| **Code, commit, repeat** | `git add … && git commit -m "feat: …"` | Use conventional commit messages; the linter will warn otherwise. |
| **Run the test‑suite** | `pytest` | Tests live in `puma/tests/`; add or extend as needed. |
| **Push & open a PR** | `git push -u origin feature/your-topic` | Mark it *Draft* if not ready. CI runs automatically. |
| **Address review feedback** | — | We squash‑merge once green. |

---

## Style & static checks

* **Formatting:** [Black](https://black.readthedocs.io/)
* **Linting & simple refactors:** [Ruff](https://docs.astral.sh/ruff/)
* **Type safety:** [mypy](https://mypy-lang.org/)
* **Import order:** isort (via ruff)

All tools are wired into [pre‑commit](https://pre-commit.com/) and enforced in CI. Running `pre-commit run --all-files` locally before pushing will save you a round‑trip. You can also enable it directly when committing by running `pre-commit install` in the puma folder. This will setup `pre-commit` in a way, that it runs everytime you commit.

---

## Reporting issues

Found a bug or have an idea? Search the [issue tracker](https://github.com/umami-hep/puma/issues) first; if it’s new, open an issue using the appropriate template. Please include:

* **Steps to reproduce**
* **Expected vs. actual behaviour**
* **Environment** (OS, Python, puma version)

---

## Communication channels

* **GitHub Issues / Pull Requests** — canonical record of work
* **Mattermost** `Umami / puma / upp` — quick questions, brainstorming

When in doubt, open an issue and tag the @puma‑maintainers.

---

## License and certificate of origin

puma is released under the MIT license. By submitting code you agree that it can be distributed under the same terms.

Happy coding — and may your $b$‑tagging be ever accurate! ✨

