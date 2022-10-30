FROM python:3.10-slim-buster

WORKDIR /qc

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
