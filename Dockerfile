FROM python:3.12-slim

RUN apt-get update && apt-get install -y git

WORKDIR /app

COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /app

CMD ["python", "/app/app.py"]