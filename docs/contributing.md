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

### Documentation Notes

- Each pull request is tested to ensure it can successfully build the documentation, preventing potential errors.
- Merging into the main branch triggers a deployment of a documentation preview, accessible at [preview.timecopilot.dev](https://preview.timecopilot.dev).
- When a new version of the library is released, the documentation is deployed to [timecopilot.dev](https://timecopilot.dev).

## Adding New Datasets

The datasets utilized in our documentation are hosted on AWS at `https://timecopilot.s3.amazonaws.com/public/data/`. If you wish to contribute additional datasets for your changes, please contact [@AzulGarza](http://github.com/AzulGarza) for guidance.