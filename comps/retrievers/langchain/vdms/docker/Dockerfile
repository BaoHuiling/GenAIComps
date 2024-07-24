# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.9-slim

ENV LANG=C.UTF-8

RUN apt-get update -y && apt-get install -y --no-install-recommends --fix-missing \
    build-essential \
    libgl1-mesa-glx \
    libjemalloc-dev \
    ffmpeg \
    vim

RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user:user /home/user/

USER user

# COPY --chown=user:user comps /home/user/comps
COPY --chown=user:user comps/retrievers/langchain/vdms/requirement.txt /home/user/comps/retrievers/langchain/vdms/requirement.txt
RUN pip install --upgrade pip setuptools &&\
    pip install -r /home/user/comps/retrievers/langchain/vdms/requirement.txt

# install customized langchain-community (VDMS)
COPY --chown=user:user comps/retrievers/langchain/vdms/langchain-community.tar.gz /home/user/comps/retrievers/langchain/vdms/langchain-community.tar.gz
RUN tar -zxf /home/user/comps/retrievers/langchain/vdms/langchain-community.tar.gz -C /home/user/comps/retrievers/langchain/vdms && \
    cd /home/user/comps/retrievers/langchain/vdms/community && \
    pip install -e .

COPY --chown=user:user comps /home/user/comps
# RUN mkdir -p /home/user/comps/retrievers/langchain/vdms/frame_metadata && chown -R user /home/user/comps/retrievers/langchain/vdms/frame_metadata

ENV PYTHONPATH=$PYTHONPATH:/home/user


WORKDIR /home/user/comps/retrievers/langchain/vdms

ENTRYPOINT ["python", "retriever_vdms.py"]