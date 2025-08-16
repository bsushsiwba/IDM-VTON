from PIL import Image
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler, AutoencoderKL
from typing import List

import torch
import os
from transformers import AutoTokenizer
import numpy as np
from gradio_demo.utils_mask import get_mask_location
from torchvision import transforms
import gradio_demo.apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import (
    convert_PIL_to_numpy,
    _apply_exif_orientation,
)
from torchvision.transforms.functional import to_pil_image
import time


device = "cuda:0" if torch.cuda.is_available() else "cpu"


import cv2
import numpy as np
from PIL import Image

print("IDM ready")

import os
import torch
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image

import os

os.system("pwd")

print("CAT ready")
