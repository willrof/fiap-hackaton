# Dockerfile

FROM python:3.12-slim-bookworm

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

ENV FLASK_APP=predict_api.py

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "predict_api:app"]