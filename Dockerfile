# Web app container for video-to-ply (Gradio front end that submits HF Jobs).
# Built and pushed to GHCR by .github/workflows/build-containers.yml.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# System deps: none strictly required; keep minimal.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py submitter.py bootstrap.sh ./
RUN chmod +x bootstrap.sh

EXPOSE 7860

CMD ["python", "app.py"]