#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -xe

WORKPATH=$(dirname "$PWD")
ip_address=$(hostname -I | awk '{print $1}')
function build_docker_images() {
    cd $WORKPATH

    # pull vdms image
    docker pull intellabs/vdms:v2.8.0

    # build dataprep image for pgvector
    docker build -t opea/dataprep-vdms:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f $WORKPATH/comps/dataprep/vdms/langchain/docker/Dockerfile .
}

function start_service() {
    cd $WORKPATH

    docker compose -f $WORKPATH/comps/dataprep/vdms/langchain/docker/docker-compose-dataprep-vdms.yaml up -d

    sleep 10s
}

function validate_microservice() {
    URL="http://$ip_address:6007/v1/dataprep"
    curl -X 'POST' \
     -H 'Content-Type: application/json' \
     -H 'accept: application/json' \
     -d '{}' \
     $URL
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=vdms-vector-db*")
    if [[ ! -z "$cid" ]]; then docker stop $cid && docker rm $cid && sleep 1s; fi

    cid=$(docker ps -aq --filter "name=dataprep-vdms-server*")
    if [[ ! -z "$cid" ]]; then docker stop $cid && docker rm $cid && sleep 1s; fi
}

function main() {

    stop_docker

    build_docker_images
    start_service

    validate_microservice

    stop_docker
    # echo y | docker system prune

}

main