# Use specific version of nvidia cuda image
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Set shell and noninteractive environment variables
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && apt-get update -y

# Set working directory
WORKDIR /

# Update and upgrade the system packages (Worker Template)
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install --yes --no-install-recommends sudo ca-certificates git wget curl bash libgl1 libx11-6 software-properties-common ffmpeg build-essential -y &&\
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

RUN wget \
    https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py39_4.9.2-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py39_4.9.2-Linux-x86_64.sh

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# Install Python dependencies (Worker Template)
RUN pip install --upgrade pip && \
    pip install faster-whisper==0.10.0 --no-cache-dir

# Copy and run script to fetch models
# COPY ./fetch_models.py /fetch_models.py
# RUN python /fetch_models.py && \
#     rm /fetch_models.py

# Copy source code into image
# COPY src .

# Set default command
# CMD python -u /rp_handler.py