FROM python:3.13-slim

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY llmcord/ ./llmcord/
COPY config/ ./config/
COPY pyproject.toml .

RUN pip install --no-cache-dir .

# Execute as a module
CMD ["python", "-m", "llmcord"]