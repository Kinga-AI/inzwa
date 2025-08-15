FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml poetry.lock* /app/
RUN pip install poetry && poetry install --no-dev --no-root && poetry build && pip install dist/*.whl

COPY src/ /app/src/

CMD ["uvicorn", "inzwa.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
