# FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
FROM python:3.9-slim
RUN apt-get update && apt-get install -y git vim gcc g++ curl tini libenchant-2-dev
WORKDIR /workdir
COPY requirements.txt ./
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download en_core_web_lg
COPY notebooks notebooks
COPY nrcan_p2 nrcan_p2
COPY *.py ./
COPY *.sh ./

CMD ["/bin/bash"]