import sys

sys.path.append("./")
from PIL import Image
import gradio as gr
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
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import (
    convert_PIL_to_numpy,
    _apply_exif_orientation,
)
from torchvision.transforms.functional import to_pil_image
import time

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in binary_mask.shape[1]:
            if binary_mask[i, j] == True:
                mask[i, j] = 1
    mask = (mask * 255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask


base_path = "yisol/IDM-VTON"
example_path = os.path.join(os.path.dirname(__file__), "example")

unet = UNet2DConditionModel.from_pretrained(
    base_path,
    subfolder="unet",
    torch_dtype=torch.float16,
)
unet.requires_grad_(False)
tokenizer_one = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer",
    revision=None,
    use_fast=False,
)
tokenizer_two = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer_2",
    revision=None,
    use_fast=False,
)
noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

text_encoder_one = CLIPTextModel.from_pretrained(
    base_path,
    subfolder="text_encoder",
    torch_dtype=torch.float16,
)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    base_path,
    subfolder="text_encoder_2",
    torch_dtype=torch.float16,
)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    base_path,
    subfolder="image_encoder",
    torch_dtype=torch.float16,
)
vae = AutoencoderKL.from_pretrained(
    base_path,
    subfolder="vae",
    torch_dtype=torch.float16,
)

# "stabilityai/stable-diffusion-xl-base-1.0",
UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
    base_path,
    subfolder="unet_encoder",
    torch_dtype=torch.float16,
)

parsing_model = Parsing(0)
openpose_model = OpenPose(0)

UNet_Encoder.requires_grad_(False)
image_encoder.requires_grad_(False)
vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder_one.requires_grad_(False)
text_encoder_two.requires_grad_(False)
tensor_transfrom = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

pipe = TryonPipeline.from_pretrained(
    base_path,
    unet=unet,
    vae=vae,
    feature_extractor=CLIPImageProcessor(),
    text_encoder=text_encoder_one,
    text_encoder_2=text_encoder_two,
    tokenizer=tokenizer_one,
    tokenizer_2=tokenizer_two,
    scheduler=noise_scheduler,
    image_encoder=image_encoder,
    torch_dtype=torch.float16,
)
pipe.unet_encoder = UNet_Encoder


def start_tryon(
    dict,
    garm_img,
    garment_des,
    is_checked,
    is_checked_crop,
    denoise_steps,
    seed,
    selected_body_part,
):

    openpose_model.preprocessor.body_estimation.model.to(device)
    pipe.to(device)
    pipe.unet_encoder.to(device)

    garm_img = garm_img.convert("RGB").resize((768, 1024))
    human_img_orig = dict["background"].convert("RGB")

    if is_checked_crop:
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize((768, 1024))
    else:
        human_img = human_img_orig.resize((768, 1024))

    # save garment and human image
    garm_img.save("garment.png")
    human_img.save("human.png")

    if os.path.exists("complete.txt"):
        os.remove("complete.txt")

    with open("process.txt", "w") as f:
        f.write("1")

    # wait while complete.txt is not created
    while not os.path.exists("complete.txt"):
        time.sleep(0.1)

    # read cloth_u.png as garm_img
    garm_img = Image.open("cloth_u.png").convert("RGB")

    # if lower body is selected, read cloth_b.png as garm_img
    if selected_body_part == "lower_body":
        garm_img = Image.open("cloth_b.png").convert("RGB")
        human_mask = None
        try:
            human_mask = Image.open("mask_b.png").convert("RGB")
        except:
            print("mask_b.png not found")
            human_mask = None

    garm_img = garm_img.resize((768, 1024))

    if is_checked:
        keypoints = openpose_model(human_img.resize((384, 512)))
        model_parse, _ = parsing_model(human_img.resize((384, 512)))
        mask, mask_gray = get_mask_location(
            "hd", selected_body_part, model_parse, keypoints
        )
        # if selected upper body then subtract lower mask form upper
        if selected_body_part == "upper_body":
            mask_b, _ = get_mask_location("hd", "lower_body", model_parse, keypoints)
            mask = Image.fromarray(
                np.clip(
                    np.array(mask, dtype=np.uint8) - np.array(mask_b, dtype=np.uint8),
                    0,
                    255,
                ).astype(np.uint8)
            )

        # if selected lower body then or with human_mask
        if selected_body_part == "lower_body" and human_mask is not None:
            mask = Image.fromarray(
                np.clip(
                    np.array(mask, dtype=np.uint8)
                    | np.array(human_mask, dtype=np.uint8),
                    0,
                    255,
                ).astype(np.uint8)
            )
        mask = mask.resize((768, 1024))
    else:
        mask = pil_to_binary_mask(dict["layers"][0].convert("RGB").resize((768, 1024)))
        # mask = transforms.ToTensor()(mask)
        # mask = mask.unsqueeze(0)
    mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
    mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    args = apply_net.create_argument_parser().parse_args(
        (
            "show",
            "./configs/densepose_rcnn_R_50_FPN_s1x.yaml",
            "./ckpt/densepose/model_final_162be9.pkl",
            "dp_segm",
            "-v",
            "--opts",
            "MODEL.DEVICE",
            "cuda",
        )
    )
    # verbosity = getattr(args, "verbosity", None)
    pose_img = args.func(args, human_img_arg)
    pose_img = pose_img[:, :, ::-1]
    pose_img = Image.fromarray(pose_img).resize((768, 1024))

    with torch.no_grad():
        # Extract the images
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                prompt = "model is wearing " + garment_des
                negative_prompt = (
                    "monochrome, lowres, bad anatomy, worst quality, low quality"
                )
                with torch.inference_mode():
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )

                    prompt = "a photo of " + garment_des
                    negative_prompt = (
                        "monochrome, lowres, bad anatomy, worst quality, low quality"
                    )
                    if not isinstance(prompt, List):
                        prompt = [prompt] * 1
                    if not isinstance(negative_prompt, List):
                        negative_prompt = [negative_prompt] * 1
                    with torch.inference_mode():
                        (
                            prompt_embeds_c,
                            _,
                            _,
                            _,
                        ) = pipe.encode_prompt(
                            prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                            negative_prompt=negative_prompt,
                        )

                    pose_img = (
                        tensor_transfrom(pose_img)
                        .unsqueeze(0)
                        .to(device, torch.float16)
                    )
                    garm_tensor = (
                        tensor_transfrom(garm_img)
                        .unsqueeze(0)
                        .to(device, torch.float16)
                    )
                    generator = (
                        torch.Generator(device).manual_seed(seed)
                        if seed is not None
                        else None
                    )
                    images = pipe(
                        prompt_embeds=prompt_embeds.to(device, torch.float16),
                        negative_prompt_embeds=negative_prompt_embeds.to(
                            device, torch.float16
                        ),
                        pooled_prompt_embeds=pooled_prompt_embeds.to(
                            device, torch.float16
                        ),
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(
                            device, torch.float16
                        ),
                        num_inference_steps=denoise_steps,
                        generator=generator,
                        strength=1.0,
                        pose_img=pose_img.to(device, torch.float16),
                        text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
                        cloth=garm_tensor.to(device, torch.float16),
                        mask_image=mask,
                        image=human_img,
                        height=1024,
                        width=768,
                        ip_adapter_image=garm_img.resize((768, 1024)),
                        guidance_scale=2.0,
                    )[0]

    if is_checked_crop:
        out_img = images[0].resize(crop_size)
        human_img_orig.paste(out_img, (int(left), int(top)))
        return human_img_orig, mask_gray
    else:
        return images[0], mask_gray
    # return images[0], mask_gray


garm_list = os.listdir(os.path.join(example_path, "cloth"))
garm_list_path = [os.path.join(example_path, "cloth", garm) for garm in garm_list]

human_list = os.listdir(os.path.join(example_path, "human"))
human_list_path = [os.path.join(example_path, "human", human) for human in human_list]

human_ex_list = []
for ex_human in human_list_path:
    ex_dict = {}
    ex_dict["background"] = ex_human
    ex_dict["layers"] = None
    ex_dict["composite"] = None
    human_ex_list.append(ex_dict)

##default human


image_blocks = gr.Blocks().queue()
with image_blocks as demo:
    gr.Markdown("## Vesti Demo")
    with gr.Row():
        with gr.Column():
            imgs = gr.ImageEditor(
                sources="upload",
                type="pil",
                label="Human. Mask with pen or use auto-masking",
                interactive=True,
            )
            is_checked = gr.State(value=True)  # Hidden state variable
            is_checked_crop = gr.State(value=True)  # Hidden state variable
            # Hidden advanced settings
            denoise_steps = gr.State(value=30)
            seed = gr.State(value=42)
            with gr.Row():
                body_part = gr.Dropdown(
                    choices=["upper_body", "lower_body", "dresses"],
                    value="upper_body",
                    label="Select Body Part",
                    info="Choose the type of clothing",
                )

            example = gr.Examples(
                inputs=imgs, examples_per_page=10, examples=human_ex_list
            )

        with gr.Column():
            garm_img = gr.Image(label="Garment", sources="upload", type="pil")
            with gr.Row(elem_id="prompt-container"):
                with gr.Row():
                    prompt = gr.Textbox(
                        placeholder="Description of garment ex) Short Sleeve Round Neck T-shirts",
                        show_label=False,
                        elem_id="prompt",
                    )
            example = gr.Examples(
                inputs=garm_img, examples_per_page=8, examples=garm_list_path
            )
        with gr.Column():
            # image_out = gr.Image(label="Output", elem_id="output-img", height=400)
            masked_img = gr.Image(
                label="Masked image output",
                elem_id="masked-img",
                show_share_button=False,
            )
        with gr.Column():
            # image_out = gr.Image(label="Output", elem_id="output-img", height=400)
            image_out = gr.Image(
                label="Output", elem_id="output-img", show_share_button=False
            )

    with gr.Column():
        try_button = gr.Button(value="Try-on")

    try_button.click(
        fn=start_tryon,
        inputs=[
            imgs,
            garm_img,
            prompt,
            is_checked,
            is_checked_crop,
            denoise_steps,
            seed,
            body_part,
        ],
        outputs=[image_out, masked_img],
        api_name="tryon",
    )


image_blocks.launch(share=True)
