FROM tensorflow/tensorflow:2.6.1-gpu

RUN apt-get update && apt-get install -y --no-install-recommends \
         git \
         libsm6 \
         libxrender1 \
         libxext6 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/
RUN pip install --upgrade pip
RUN pip install --requirement /tmp/requirements.txt
RUN pip install pip install git+https://github.com/google-research/meta-dataset.git

WORKDIR /workspace
