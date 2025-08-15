

CORE_REF ?= v0.1.0
CORE_DIR ?= ../kinga-core

sync-core:
\t@if [ ! -d $(CORE_DIR) ]; then echo 'missing core'; exit 1; fi
\tcp $(CORE_DIR)/lang/registry.yaml ./langpacks/REGISTRY.yaml

install:
	poetry install  # Or pip install -e .

test:
	pytest tests/

run:
	poetry run uvicorn inzwa.api.app:app --host 0.0.0.0 --port 8000

docker-build:
	docker build -t inzwa:latest .  # Assumes Dockerfile at root

docker-run:
	docker run -p 8000:8000 inzwa:latest

