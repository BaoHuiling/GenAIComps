
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM python:3.9-slim

ENV LANG=C.UTF-8

RUN apt-get update -y && apt-get install -y --no-install-recommends --fix-missing \
    vim

RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user:user /home/user/

USER user

# COPY --chown=user:user comps /home/user/comps
COPY --chown=user:user comps/dataprep/vdms/langchain/requirements.txt /home/user/comps/dataprep/vdms/langchain/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -r /home/user/comps/dataprep/vdms/langchain/requirements.txt


COPY --chown=user:user comps /home/user/comps

ENV PYTHONPATH=/home/user


WORKDIR /home/user/comps/dataprep/vdms/langchain
RUN pip install uvicorn

ENTRYPOINT ["bash","start.sh"]
