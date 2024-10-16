# FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
FROM python:3.9-slim
RUN apt-get update && apt-get install -y git vim gcc g++ curl tini libenchant-2-dev libgtk-3-dev
WORKDIR /workdir

# COPY requirements.txt ./
# RUN pip install -r requirements.txt
# RUN python -m spacy download en_core_web_sm
# RUN python -m spacy download en_core_web_lg
# COPY notebooks notebooks
# COPY nrcan_p2 nrcan_p2
# COPY *.py ./
# COPY *.sh ./
# CMD ["/bin/bash"]

RUN git clone https://github.com/DARPA-CRITICALMAAS/sri-ta2-mappable-criteria.git
RUN pip3 install -r /workdir/sri-ta2-mappable-criteria/polygon_ranking/requirements.txt
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
WORKDIR sri-ta2sri-ta2-mappable-criteria
ENTRYPOINT ["streamlit", "run", "Welcome.py", "--server.port=8501", "--server.address=0.0.0.0"]