.PHONY: install install-full run test clean

## Create venv and install all core dependencies
install:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e .
	@echo ""
	@echo "✅ Core install complete. Run: source .venv/bin/activate && ms-rag"

## Install everything including all extras (vector DBs, evaluators, rerankers)
install-full:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e .
	@echo ""
	@echo "✅ Full production install complete. Run: source .venv/bin/activate && ms-rag"

## Run the framework
run:
	.venv/bin/ms-rag

## Run all tests
test:
	.venv/bin/pytest tests/ -v

## Clean venv and cache
clean:
	rm -rf .venv __pycache__ .pytest_cache .hypothesis chroma_db
