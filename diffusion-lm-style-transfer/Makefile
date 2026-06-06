.PHONY: install install-dev lint format typecheck test test-cov clean

install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev,notebooks]"
	pre-commit install

lint:
	ruff check .

format:
	black .
	ruff check --fix .

typecheck:
	mypy phase1_embedding_foundation phase2_forward_diffusion \
	     phase3_denoising_network phase4_style_conditioning \
	     phase5_style_transfer_inference phase6_training_pipeline \
	     phase7_evaluation

test:
	pytest

test-cov:
	pytest --cov=. --cov-report=html --cov-report=term-missing

pretrain:
	python phase6_training_pipeline/pretrain.py

train-classifier:
	python phase6_training_pipeline/train_classifier.py

finetune:
	python phase6_training_pipeline/finetune.py

evaluate:
	python phase7_evaluation/evaluate.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	rm -f .coverage
