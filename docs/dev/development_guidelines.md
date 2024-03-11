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
