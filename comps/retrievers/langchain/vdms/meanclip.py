import logging
from typing import List, Dict, Any

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