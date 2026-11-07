# Container for the API. Build and run:
#   docker build -t vayucast .
#   docker run -p 8000:8000 vayucast
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY api ./api
COPY models ./models
COPY reports ./reports

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
