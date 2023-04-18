FROM python:3.8.15-slim

RUN apt update && apt install -y vim

# install requirements that are needed for development
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the current version of the repo in the image
COPY . /puma_repo

# Install and remove the folder afterwards
RUN pip install /puma_repo && rm -rf /puma_repo
