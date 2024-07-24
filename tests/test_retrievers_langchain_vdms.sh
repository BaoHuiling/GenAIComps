#!/bin/bash
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -xe

WORKPATH=$(dirname "$PWD")
LOG_PATH="$WORKPATH/tests"
ip_address=$(hostname -I | awk '{print $1}')

function build_docker_images() {
    cd $WORKPATH
    docker build -t opea/retriever-vdms:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/retrievers/langchain/vdms/docker/Dockerfile .
}

function start_service() {
    cd $WORKPATH
    export VDMS_URL="http://${ip_address}:55555"
    
    # vdms
    docker run -d --name="vdms-vector-db" -p 55555:55555 intellabs/vdms:latest
    sleep 10s

    # retriever
    docker compose -f $WORKPATH/comps/retrievers/langchain/vdms/docker/docker_compose_retriever.yaml up -d
    sleep 1m
}

function validate_microservice() {
    retriever_port=7000
    URL="http://${ip_address}:$retriever_port/v1/retrieval"

    # retrieve
    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST -d "{\"text\":\"A man shopping\"}" -H 'Content-Type: application/json' "$URL")
    if [ "$HTTP_STATUS" -eq 200 ]; then
        echo "[ retriever ] HTTP status is 200. Checking content..."
        local CONTENT=$(curl -s -X POST -d "{\"text\":\"A man shopping\"}" -H 'Content-Type: application/json' "$URL" | tee ${LOG_PATH}/retriever.log)

        if echo "$CONTENT" | grep -q "retrieved_docs"; then
            echo "[ retriever ] Content is as expected."
        else
            echo "[ retriever ] Content does not match the expected result: $CONTENT"
            docker logs retriever-vdms-server >> ${LOG_PATH}/retriever.log
            exit 1
        fi
    else
        echo "[ retriever ] HTTP status is not 200. Received status was $HTTP_STATUS"
        docker logs retriever-vdms-server >> ${LOG_PATH}/retriever.log
        exit 1
    fi
}

function stop_docker() {
    cid=$(docker ps -aq --filter "name=vdms-vector-db*")
    if [[ ! -z "$cid" ]]; then docker stop $cid && docker rm $cid && sleep 1s; fi

    cid=$(docker ps -aq --filter "name=retriever-vdms-server*")
    if [[ ! -z "$cid" ]]; then docker stop $cid && docker rm $cid && sleep 1s; fi
}

function main() {

    stop_docker

    build_docker_images
    start_service

    validate_microservice

    stop_docker
    echo y | docker system prune

}

main
