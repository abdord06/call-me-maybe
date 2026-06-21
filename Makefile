.PHONY: install run debug clean lint lint-strict

# Install dependencies using uv
install:
	uv sync

# Run the main script
run:
	uv run python -m src

# Run the main script in debug mode using pdb
debug:
	uv run python -m pdb -m src

# Clean temporary files and caches
clean:
	rm -rf __pycache__
	rm -rf .mypy_cache
	rm -rf src/__pycache__
	rm -rf llm_sdk/__pycache__

# Lint the code using flake8 and mypy with specific flags
lint:
	uv run flake8 src/
	uv run mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs src/

# Strict linting
lint-strict:
	uv run flake8 src/
	uv run mypy --strict src/
