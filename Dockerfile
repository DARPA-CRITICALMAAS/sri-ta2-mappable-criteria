FROM ubuntu:24.04

RUN apt update
RUN apt install -y python3-pip python3.12-venv
RUN apt-get update
RUN apt-get install -y \
    libgtk-3-dev \
    libgdal-dev \
    gdal-bin \
    python3-gdal \
    python3.12-dev \
    unzip
RUN export CPLUS_INCLUDE_PATH=/usr/include/gdal
RUN export C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /home/ubuntu/app/sri-ta2-mappable-criteria
COPY . .

RUN python3 -m venv /home/ubuntu/venvs/sri-map-synth
ENV VIRTUAL_ENV /home/ubuntu/venvs/sri-map-synth
ENV PATH /home/ubuntu/venvs/sri-map-synth/bin:$PATH
RUN pip install GDAL==`gdal-config --version`
RUN pip install -r polygon_ranking/requirements.txt

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit", "run", "Welcome.py", "--server.port=8501", "--server.address=0.0.0.0"]