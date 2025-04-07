FROM python:3.11.10-bullseye

RUN apt update && apt install -y vim nano

# install requirements that are needed for development
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# copy and install package
COPY . .
RUN python -m pip install -e .

