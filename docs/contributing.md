Your contributions are highly appreciated!

## Installation and Setup

Clone your fork and cd into the repo directory

```bash
git clone git@github.com:<your username>/TimeCopilot.git
cd TimeCopilot
```

Install `uv`, and `pre-commit`:

* [`uv` install docs](https://docs.astral.sh/uv/getting-started/installation/)
* [`pre-commit` install docs](https://pre-commit.com/#install)

!!! tip
    Once `uv` is installed, to install `pre-commit` you can run the following command:

    ```bash
    uv tool install pre-commit
    ```

Install the required libraries for local development

```bash
uv sync --frozen --all-extras --all-packages --group docs
```

Install `pre-commit` hooks

```bash
pre-commit install --install-hooks
```

You're ready to start contributing! 

## Running Tests

To run tests, run:

```bash
uv run pytest
```

## Documentation Changes

To run the documentation page locally, run:

```bash
uv run mkdocs serve
```