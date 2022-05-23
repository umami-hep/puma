# Good coding practices

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
`flake8` and `pylint` to check the code for bad coding practices. Make sure to run them before
you commit your code.

In addition to that, we require docstrings in the `numpy`-style, which are checked in
the pipeline by `darglint`.

## Pre-commit hook

To check staged files for style and `flake8` conformity, you can use the `pre-commit`
hook, which then won't allow you to commit your staged changes if `isort`, `black` or
`flake8` fails.
You might have to set it up by executing the following in the root of the repo:

```bash
pre-commit install
```
