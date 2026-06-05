# How to contribute

## Getting started

This project uses [uv](https://docs.astral.sh/uv/) for environment management.
After cloning, create the environment and install the project with all dev
dependencies:

```bash
uv sync
```

This installs RIANN in editable mode along with the test, docs, and example
toolchains.

## Running the checks

```bash
uv run pytest                  # run the test suite
uv run ruff check .            # lint
uv run ruff format .           # auto-format (use --check in CI)
```

CI runs the same `ruff check`, `ruff format --check`, and `pytest` on Python
3.9–3.12, so please make sure they pass locally before opening a PR.

## Example notebooks

The notebooks in `examples/` are demonstrations, not part of the library. Each
notebook is paired with a percent-format `.py` script via
[jupytext](https://jupytext.readthedocs.io/) so the code diffs cleanly in
version control. After editing either side, keep them in sync:

```bash
uv run jupytext --sync examples/<notebook>.ipynb
```

To regenerate a notebook's saved outputs:

```bash
uv run jupytext --to ipynb --execute examples/<notebook>.py
```

The `examples/` directory is excluded from Ruff (jupytext owns its layout).

## Documentation

Docs are built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/)
and [mkdocstrings](https://mkdocstrings.github.io/); the API reference is
generated from the NumPy-style docstrings in `riann/riann.py`. Preview locally:

```bash
uv run mkdocs serve
```

The site is published to GitHub Pages automatically on pushes to `master`.

## Did you find a bug?

* Ensure the bug was not already reported by searching on GitHub under Issues.
* If you're unable to find an open issue addressing the problem, open a new one.
  Include a title, a clear description, and a minimal code sample or test case
  that demonstrates the problem.

## Did you write a patch that fixes a bug?

* Open a new GitHub pull request with the patch.
* Ensure your PR includes a test that fails without the patch and passes with it.
* Ensure the PR description clearly describes the problem and solution, and
  references the relevant issue number if applicable.

## PR submission guidelines

* Keep each PR focused on a single concern.
* Do not mix style-only changes with functional changes.
