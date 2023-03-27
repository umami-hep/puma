FROM python:3.8.15-slim

# Copy the current version of the repo in the image
COPY . /puma_repo
# Install and remove the folder afterwards
RUN pip install /puma_repo && rm -rf /puma_repo
RUN pip install h5py==3.8.0