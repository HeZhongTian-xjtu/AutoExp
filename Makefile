.PHONY: install lint test web docker-build compose-up package-check

PYTHON ?= python

install:
	$(PYTHON) -m pip install -r requirements.txt

lint:
	$(PYTHON) -m ruff check autoexp scripts
	$(PYTHON) -m black --check autoexp scripts

test:
	$(PYTHON) -m pytest -q

package-check:
	$(PYTHON) -m build
	$(PYTHON) -m twine check dist/*

web:
	$(PYTHON) -m streamlit run autoexp/webui/app.py

docker-build:
	docker build -f Dockerfile.autoexp -t autoexp-runner:latest .

compose-up:
	docker compose up -d --build
