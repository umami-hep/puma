# Good coding practices

## Development Setup

### Installing for development

For development, we recommend using [`uv`](https://docs.astral.sh/uv/), a fast Python package installer and resolver written in Rust. It provides significantly faster dependency resolution and installation compared to traditional tools.

#### Install uv

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip (If installing from PyPI, we recommend installing uv into an isolated environment)
pip install uv
```

#### Install puma in development mode

```bash
git clone https://github.com/umami-hep/puma.git
cd puma
uv sync --extra dev
```

This installs `puma` in editable mode with all development dependencies including testing frameworks, linters, formatters, etc.

#### Alternative: Using pip

If you prefer to use `pip`, you can still install with:

```bash
python -m pip install -e ".[dev]"
```

## Test-Driven Development

The `puma` framework uses unit tests to reduce the risk for bugs being undetected.
If you contribute to `puma`, please make sure that you add unit tests for the new
code.

## Code Style

We are using the [`black`](https://github.com/psf/black) python formatter, which
also runs in the pipeline to check if your code is properly formatted.
Most editors have a quite nice integration of `black` where you can e.g. set up
automatic formatting when you save a file.

## Linters

In addition to the pure style-component of checking the code with `black`, we use
`ruff` to check the code for bad coding practices and docstrings. Make sure to run 
`ruff` before you commit your code.

## Pre-commit hook

To check staged files for style conformity, you can use the `pre-commit`
hook, which then won't allow you to commit your staged changes if `ruff` 
or `black fails.
You can set it up by executing the following in the root of the repo:

```bash
pre-commit install
```
