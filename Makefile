.PHONY: install train api test

install:
	pip install -r requirements.txt

train:
	jupyter nbconvert --to notebook --execute notebooks/fraud_detection_v2.ipynb

api:
	uvicorn src.api.main:app --reload --port 8000

test:
	pytest tests/ -v
