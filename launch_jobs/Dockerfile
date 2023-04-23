FROM python:3.8-slim

COPY ./launch_jobs/requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY ./launch_jobs /anomalib
WORKDIR /anomalib

CMD ["python", "/anomalib/train.py"]