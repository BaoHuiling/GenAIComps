# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from typing import Optional

from pydantic import BaseModel, Field
from langchain_community.vectorstores import VDMS # type: ignore
from langchain_community.vectorstores.vdms import VDMS_Client # type: ignore

from config import COLLECTION_NAME, VDMS_HOST, VDMS_PORT, MEANCLIP_CFG
from comps import opea_microservices, register_microservice
from extract_store_frames import process_all_videos
from generate_store_embeddings import setup_meanclip_model, MeanCLIPEmbeddings

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:     [%(asctime)s] %(message)s",
    datefmt="%d/%m/%Y %I:%M:%S"
    )

class ConfigDoc(BaseModel):
    video_folder: Optional[str] = Field("./videos", description="Path to folder containing videos to upload.")
    chunck_duration: Optional[int] = Field(30, description="Duration in seconds of each video segment.")
    clip_duration: Optional[int] = Field(10, description="Duration in seconds of the initial segment used for embedding calculation from each chunck.")

def read_json(path):
    with open(path) as f:
        x = json.load(f)
    return x

def ingest_files_to_vdms(metadata_file_path):
    
    GMetadata = read_json(metadata_file_path)
    global_counter = 0

    total_videos = len(GMetadata.keys())
    
    # Create vdms client
    logging.info('Connecting to VDMS db server . . .')
    client = VDMS_Client(host=VDMS_HOST, port=VDMS_PORT)
    
    # create embeddings using local embedding model
    model, _ = setup_meanclip_model(MEANCLIP_CFG, device="cpu")
    embedder = MeanCLIPEmbeddings(model=model)
    
    # create vectorstore
    vectorstore = VDMS(
                    client = client,
                    embedding = embedder,
                    collection_name = COLLECTION_NAME,
                    engine = "FaissFlat",
                    distance_strategy="IP"
                )
    
    for idx, (video, data) in enumerate(GMetadata.items()):

        metadata_list = []
        
        # process frames
        try:
            data['video'] = video
            video_name_list = [data["video_path"]]
            metadata_list = [data]
            vectorstore.add_videos(
                paths=video_name_list,
                metadatas=metadata_list,
                start_time=[data['timestamp']],
                clip_duration=[data['clip_duration']]
            )
        except Exception as e:
            logging.error(e)
            
        logging.info(f"✅ {idx+1}/{total_videos} video {video}, {len(video_name_list)} videos clips, {len(metadata_list)} metadatas")
        
@register_microservice(
    name="opea_service@prepare_doc_vdms",
    endpoint="/v1/dataprep",
    host="0.0.0.0",
    port=6007
)
async def ingest_documents(input: Optional[ConfigDoc]):
    Optional
    meta_output_dir = "./frame_metadata"
    video_folder = input.video_folder
    chunck_duration = input.chunck_duration
    clip_duration = input.clip_duration
    
    logging.info( f"video_folder: {video_folder}, meta_output_dir: {meta_output_dir}, chunck_duration: {chunck_duration}, clip_duration: {clip_duration}")
    
    process_all_videos(video_folder, meta_output_dir, chunck_duration, clip_duration) # extract frames
    global_metadata_file_path = meta_output_dir + "/metadata.json"
    ingest_files_to_vdms(global_metadata_file_path)

    logging.info(f"Successfully saved videos under {video_folder}")
    return {"status": 200, "message": "Data preparation succeeded"}

if __name__ == "__main__":
    opea_microservices["opea_service@prepare_doc_vdms"].start()
