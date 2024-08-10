# Dataprep Microservice with VDMS

# 🚀Start Microservice with Docker

## 1. Setup Environment Variables

```bash
export host_ip=${your_host_ip}
export VDMS_URL="http://${host_ip}:55555"
export http_proxy=${your_http_proxy}
export https_proxy=${your_http_proxy}
export no_proxy=${no_proxy},${host_ip}
export COLLECTION_NAME=${your_collection_name} # optional
```

## 2. Build Docker Image

```bash
docker build -t opea/dataprep-vdms:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/dataprep/vdms/langchain/docker/Dockerfile .
```

## 3 Start the service

```bash
docker compose -f comps/dataprep/vdms/langchain/docker/docker-compose-dataprep-vdms.yaml up -d
```

## 4 Ingest videos

To use customized videos, please add to video folder `comps/dataprep/vdms/langchain/videos`, it will be mounted to the container.

```bash
ip_address=$(hostname -I | awk '{print $1}')
curl -X 'POST' \
     -H 'Content-Type: application/json' \
     -H 'accept: application/json' \
     -d '{}' \
     "http://${ip_address}:6007/v1/dataprep"
```

Configurable parameter:
- video_folder: Path to folder containing videos to upload. Default: ./videos
- chunk_duration: Duration in seconds of each video segment. Default: 30s
- clip_duration: Duration in seconds of the initial segment used for embedding calculation from each chunk. Default: 10s

```bash
curl -X 'POST' \
     -H 'Content-Type: application/json' \
     -H 'accept: application/json' \
     -d '{ "video_folder": "./videos", "chunk_duration": 30, "clip_duration": 10}' \
     "http://${ip_address}:6007/v1/dataprep"
```
