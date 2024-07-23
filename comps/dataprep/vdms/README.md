# Dataprep Microservice with VDMS

# 🚀Start Microservice with Docker

## Setup Environment Variables

```bash
export host_ip=${your_host_ip}
export VDMS_URL="http://${host_ip}:55555"
export COLLECTION_NAME=${your_collection_name} # optional
export http_proxy=${your_http_proxy}
export https_proxy=${your_http_proxy}
export no_proxy=${no_proxy},${host_ip}
```

## Build Docker Image

```bash
docker build -t opea/dataprep-vdms:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f comps/dataprep/vdms/langchain/docker/Dockerfile .
```

## Start the service

```bash
docker compose -f comps/dataprep/vdms/langchain/docker/docker-compose-dataprep-vdms.yaml up -d
```

### Ingest videos 

```bash
# use default video folder: videos
curl -X 'POST' \
     -H 'Content-Type: application/json' \
     -H 'accept: application/json' \
     -d '{}' \
     'http://{host_ip}:6007/v1/dataprep'

# use customized folder
#   - please prepare the videos before building the image
#   - or mount the folder in docker compose yaml
# video_folder: Path to folder containing videos to upload.
# chunck_duration: Duration in seconds of each video segment.
# clip_duration: Duration in seconds of the initial segment used for embedding calculation from each chunck.
curl -X 'POST' \
     -H 'Content-Type: application/json' \
     -H 'accept: application/json' \
     -d '{ "video_folder": "./videos", "chunck_duration": 30, "clip_duration": 10}' \
     'http://{host_ip}:6007/v1/dataprep'
```
