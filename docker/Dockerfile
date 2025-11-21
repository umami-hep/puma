FROM python:3.11.10-bullseye

RUN apt update && apt install -y vim nano

# copy and install package
COPY . .
RUN python -m pip install -e .[dev]

