.PHONY: serve-docs
serve-docs:
	uv run --group docs mkdocs build
	uv run --group docs modal serve .github/deploy_docs.py