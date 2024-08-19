# FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
FROM python:3.9-slim
RUN apt-get update && apt-get install -y git vim gcc g++ curl tini
COPY requirements.txt ./
RUN pip install -r requirements.txt
RUN FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install --upgrade flash-attn --no-build-isolation
RUN python -m nltk.downloader popular
# copy stuff
WORKDIR /workdir
COPY configs configs
COPY *.py ./
COPY *.csv ./

# CMD ["/bin/bash"]
CMD ["python", "main.py", $FOMMA_CONFIG_FILE]

