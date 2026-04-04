# checkers/logreg-digits/Dockerfile
FROM python:3.12-slim

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY check.py .
ENTRYPOINT ["python", "check.py"]