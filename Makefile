.PHONY: help install test clean lint format setup data train
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

setup: ## Set up the development environment
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt
	. venv/bin/activate && pip install -e .
	cp .env.example .env

install: ## Install dependencies
	pip install -r requirements.txt
	pip install -e .

test: ## Run tests
	pytest tests/ -v

test-coverage: ## Run tests with coverage
	pytest --cov=src --cov-report=html tests/

lint: ## Run linting
	flake8 src/ tests/
	pylint src/

format: ## Format code
	black src/ tests/
	isort src/ tests/

clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf build/
	rm -rf dist/

data: ## Process vehicle listing data
	python -c "from src.data.data_loader import load_raw_data; print('Lead generation data processing script here')"

train: ## Train lead prediction models
	python -c "from src.models.model_trainer import ModelTrainer; print('Lead prediction model training script here')"

jupyter: ## Start Jupyter notebook
	jupyter notebook notebooks/

docs: ## Generate documentation
	sphinx-build -b html docs/ docs/_build/
