FROM python:3.11-slim

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -e .

ENV SEIGE_TARGET_BACKEND=mock
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
