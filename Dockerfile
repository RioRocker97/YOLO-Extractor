
FROM python:3.10-slim

ENV PYTHONUNBUFFERED True
ENV PYTHONDONTWRITEBYTECODE True

WORKDIR /app
COPY . ./

RUN apt-get update; apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --no-cache-dir -r requirements.txt

CMD exec gunicorn --bind :$PORT --workers 1 -k uvicorn.workers.UvicornWorker --threads 8 --timeout 0 photo_api:ASGI_APP
