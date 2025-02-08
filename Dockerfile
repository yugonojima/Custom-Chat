FROM python:3.10-bullseye

WORKDIR /custom-chat

RUN apt-get update && apt-get -y upgrade

COPY . /custom-chat
EXPOSE 8501

RUN pip install -r ./requirements-dev.txt
