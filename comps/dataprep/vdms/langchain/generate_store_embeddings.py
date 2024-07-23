# # import sys
# # import os

# # sys.path.append('/path/to/parent')  # Replace with the actual path to the parent folder

# # from VideoRAGQnA.utils import config_reader as reader
# import sys
# import os

# # Add the parent directory of the current script to the Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# VECTORDB_SERVICE_HOST_IP = os.getenv("VECTORDB_SERVICE_HOST_IP", "0.0.0.0")


# # sys.path.append(os.path.abspath('../utils'))
# # import config_reader as reader
# import yaml
# import json
# import os
# import argparse
# from langchain_experimental.open_clip import OpenCLIPEmbeddings

import logging
from typing import List, Optional, Iterable, Dict, Any

# package imports
from decord import VideoReader, cpu
from einops import rearrange
from langchain.pydantic_v1 import BaseModel, root_validator
from langchain_core.embeddings import Embeddings
import numpy as np
from PIL import Image
import torch

# custom imports
from embedding import MeanCLIP, CLIP, get_transforms, SimpleTokenizer


# from utils import config_reader as reader
# from embedding.extract_store_frames import process_all_videos
# from embedding.vector_stores import db
# from decord import VideoReader, cpu
# import numpy as np
# from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:     [%(asctime)s] %(message)s",
    datefmt="%d/%m/%Y %I:%M:%S"
    )


def setup_meanclip_model(cfg, device):

    pretrained_state_dict = CLIP.get_config(pretrained_clip_name=cfg.clip_backbone)
    state_dict = {}
    epoch = 0
    logging.info("Loading CLIP pretrained weights ...")
    for key, val in pretrained_state_dict.items():    
        new_key = "clip." + key
        if new_key not in state_dict:
            state_dict[new_key] = val.clone()

    if cfg.sim_header != "meanP":
        for key, val in pretrained_state_dict.items():
            # initialize for the frame and type postion embedding
            if key == "positional_embedding":
                state_dict["frame_position_embeddings.weight"] = val.clone()

            # using weight of first 4 layers for initialization
            if key.find("transformer.resblocks") == 0:
                num_layer = int(key.split(".")[2])

                # initialize the 4-layer temporal transformer
                if num_layer < 4:
                    state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                    continue

                if num_layer == 4: # for 1-layer transformer sim_header
                    state_dict[key.replace(str(num_layer), "0")] = val.clone()

    model = MeanCLIP(cfg, pretrained_state_dict)
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='')

    if str(device) == "cpu":
        model.float()

    model.to(device)

    logging.info("Setup model done!")
    return model, epoch

class MeanCLIPEmbeddings(BaseModel, Embeddings):
    """MeanCLIP Embeddings model."""

    model: Any
    preprocess: Any
    tokenizer: Any
    # Select model: https://github.com/mlfoundations/open_clip
    model_name: str = "ViT-H-14"
    checkpoint: str = "laion2b_s32b_b79k"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that open_clip and torch libraries are installed."""
        try:
            # Use the provided model if present
            if "model" not in values:
                raise ValueError("Model must be provided during initialization.")
            values["preprocess"] = get_transforms
            values["tokenizer"] = SimpleTokenizer()

        except ImportError:
            raise ImportError(
                "Please ensure CLIP model is loaded"
            )
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        model_device = next(self.model.clip.parameters()).device
        text_features = []
        for text in texts:
            # Tokenize the text
            if isinstance(text, str):
                text = [text]

            sot_token = self.tokenizer.encoder["<|startoftext|>"]
            eot_token = self.tokenizer.encoder["<|endoftext|>"]
            tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts]
            tokenized_text = torch.zeros((len(tokens), 64), dtype=torch.int64)
            for i in range(len(tokens)):
                if len(tokens[i]) > 64:
                    tokens[i] = tokens[i][:64-1] + tokens[i][-1:]
                tokenized_text[i, :len(tokens[i])] = torch.tensor(tokens[i])
            #print("text:", text[i])
            #print("tokenized_text:", tokenized_text[i,:10])
            text_embd, word_embd = self.model.get_text_output(tokenized_text.unsqueeze(0).to(model_device), return_hidden=False)

            # Normalize the embeddings
            #print(" --->>>> text_embd.shape:", text_embd.shape)
            text_embd = rearrange(text_embd, "b n d -> (b n) d")
            text_embd = text_embd / text_embd.norm(dim=-1, keepdim=True)

            # Convert normalized tensor to list and add to the text_features list
            embeddings_list = text_embd.squeeze(0).tolist()
            #print("text embedding:", text_embd.flatten()[:10])
            text_features.append(embeddings_list)

        return text_features


    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


    def embed_video(self, paths: List[str], **kwargs: Any) -> List[List[float]]:
        # Open images directly as PIL images

        video_features = []
        for vid_path in sorted(paths):
            # Encode the video to get the embeddings
            model_device = next(self.model.parameters()).device
            # Preprocess the video for the model
            videos_tensor= self.load_video_for_meanclip(vid_path, num_frm=self.model.num_frm,
                                                                              max_img_size=224,
                                                                              start_time=kwargs.get("start_time", None),
                                                                              clip_duration=kwargs.get("clip_duration", None)
                                                                              )
            embeddings_tensor = self.model.get_video_embeddings(videos_tensor.unsqueeze(0).to(model_device))

            # Convert tensor to list and add to the video_features list
            embeddings_list = embeddings_tensor.squeeze(0).tolist()

            video_features.append(embeddings_list)

        return video_features


    def load_video_for_meanclip(self, vis_path, num_frm=64, max_img_size=224, **kwargs):
        # Load video with VideoReader
        vr = VideoReader(vis_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        num_frames = len(vr)
        start_idx = int(fps*kwargs.get("start_time", [0])[0])
        end_idx = start_idx+int(fps*kwargs.get("clip_duration", [num_frames])[0])

        frame_idx = np.linspace(start_idx, end_idx, num=num_frm, endpoint=False, dtype=int) # Uniform sampling
        clip_images = []

        # Extract frames as numpy array
        #img_array = vr.get_batch(frame_idx).asnumpy() # img_array = [T,H,W,C]
        #clip_imgs = [Image.fromarray(img_array[j]) for j in range(img_array.shape[0])]
        # write jpeg to tmp
        import os 
        os.makedirs('tmp', exist_ok=True)
        os.system(f"ffmpeg -nostats -loglevel 0 -i {vis_path} -q:v 2 tmp/img%03d.jpeg")
        #print("vis_path:", vis_path)
        #print("frame_idx:", frame_idx)

        # preprocess images
        clip_preprocess = get_transforms("clip", max_img_size)
        for img_idx in frame_idx:
            #im = clip_imgs[i]
            im = Image.open(f'tmp/img{img_idx+1:03d}.jpeg')
            clip_images.append(clip_preprocess(im)) # 3, 224, 224

        os.system("rm -r tmp")
        clip_images_tensor = torch.zeros((num_frm,) + clip_images[0].shape)
        clip_images_tensor[:num_frm] = torch.stack(clip_images)

        return clip_images_tensor
# def read_json(path):
#     with open(path) as f:
#         x = json.load(f)
#     return x

# def read_file(path):
#     content = None
#     with open(path, 'r') as file:
#         content = file.read()
#     return content

# def store_into_vectordb(vs, metadata_file_path, config):
#     GMetadata = read_json(metadata_file_path)
#     global_counter = 0

#     total_videos = len(GMetadata.keys())
    
#     for idx, (video, data) in enumerate(GMetadata.items()): #遍历metadata.json中所有视频

#         image_name_list = []
#         embedding_list = []
#         metadata_list = []
#         ids = []
        
#         if config['embeddings']['type'] == 'frame':
#             # process frames 从文件中获取metadata的值，写到list里
#             frame_metadata = read_json(data['extracted_frame_metadata_file'])
#             for frame_id, frame_details in frame_metadata.items():
#                 global_counter += 1
#                 if vs.selected_db == 'vdms':
#                     meta_data = {
#                         'timestamp': frame_details['timestamp'],
#                         'frame_path': frame_details['frame_path'],
#                         'video': video,
#                         #'embedding_path': data['embedding_path'],
#                         'date_time': frame_details['date_time'], #{"_date":frame_details['date_time']},
#                         'date': frame_details['date'],
#                         'year': frame_details['year'],
#                         'month': frame_details['month'],
#                         'day': frame_details['day'],
#                         'time': frame_details['time'],
#                         'hours': frame_details['hours'],
#                         'minutes': frame_details['minutes'],
#                         'seconds': frame_details['seconds'],
#                     }

#                 image_path = frame_details['frame_path']
#                 image_name_list.append(image_path)

#                 metadata_list.append(meta_data)
#                 ids.append(str(global_counter))
#                 # print('datetime',meta_data['date_time'])

#             vs.add_images(
#                 uris=image_name_list,
#                 metadatas=metadata_list
#             )
#         elif config['embeddings']['type'] == 'video':
#             data['video'] = video
#             video_name_list = [data["video_path"]]
#             metadata_list = [data]
#             if vs.selected_db == 'vdms':
#                 vs.video_db.add_videos(
#                     paths=video_name_list,
#                     metadatas=metadata_list,
#                     start_time=[data['timestamp']],
#                     clip_duration=[data['clip_duration']]
#                 )
#                 # 定义add_videos, 数据已经有模型了
#             else:
#                 print(f"ERROR: selected_db {vs.selected_db} not supported. Supported:[vdms]")
#         print (f'✅ {idx+1}/{total_videos} video {video}')

# def generate_embeddings(config, vs):
#     # embedding_model没用到
#     process_all_videos(config) # 生成metadata.json
#     global_metadata_file_path = os.path.join(config["meta_output_dir"], 'metadata.json')
#     print(f'global metadata file available at {global_metadata_file_path}')
#     store_into_vectordb(vs, global_metadata_file_path, config)

# def retrieval_testing(vs):
#     Q = 'Man holding red shopping basket'
#     print (f'Testing Query: {Q}')
#     top_k = 3
#     results = vs.MultiModalRetrieval(Q, top_k=top_k)

#     print(f"top-{top_k} returned results:", results)

# def main():
#     # read config yaml
#     print ('Reading config file')
#     # config = reader.read_config('../docs/config.yaml')

#     # Create argument parser
#     parser = argparse.ArgumentParser(description='Process configuration file for generating and storing embeddings.')
#     parser.add_argument('config_file', type=str, help='Path to configuration file (e.g., config.yaml)')

#     # Parse command-line arguments
#     args = parser.parse_args()
#     # Read configuration file
#     config = reader.read_config(args.config_file)
# # Read MeanCLIP
# meanclip_cfg_json = json.load(open(config['meanclip_cfg_path'], 'r'))
# meanclip_cfg = argparse.Namespace(**meanclip_cfg_json)


#     print ('Config file data \n', yaml.dump(config, default_flow_style=False, sort_keys=False))

#     generate_frames = config['generate_frames']
#     #embed_frames = config['embed_frames']
#     path = config['videos'] #args.videos_folder #
#     meta_output_dir = config['meta_output_dir']
#     N = config['number_of_frames_per_second']
#     emb_path = config['embeddings']['path']

#     host = VECTORDB_SERVICE_HOST_IP
#     port = int(config['vector_db']['port'])
#     selected_db = config['vector_db']['choice_of_db']

#     # Creating DB
#     print ('Creating DB with video embedding and metadata support, \nIt may take few minutes to download and load all required models if you are running for first time.')
#     print('Connecting to {} at {}:{}'.format(selected_db, host, port))

#     if config['embeddings']['type'] == 'frame':
#         vs = db.VS(host, port, selected_db)
#         # EMBEDDING MODEL
#         model = OpenCLIPEmbeddings(model_name="ViT-g-14", checkpoint="laion2b_s34b_b88k")

#     elif config['embeddings']['type'] == 'video':
#         # init meanclip model
#         model, _ = setup_meanclip_model(meanclip_cfg, device="cpu")
#         vs = db.VideoVS(host, port, selected_db, model)
#     else:
#         print(f"ERROR: Selected embedding type in config.yaml {config['embeddings']['type']} is not in [\'video\', \'frame\']")
#         return
#     generate_embeddings(config, vs)
#     retrieval_testing(vs)
#     return vs

# if __name__ == '__main__':
#     main()