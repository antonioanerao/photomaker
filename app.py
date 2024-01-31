import torch
import numpy as np
import random
import os
import socket

from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler

from huggingface_hub import hf_hub_download
import spaces
import gradio as gr

from photomaker import PhotoMakerStableDiffusionXLPipeline
from style_template import styles

# global variable
base_model_path = 'SG161222/RealVisXL_V3.0'
cache_dir='cache_dir'
device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEED = np.iinfo(np.int32).max
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Photographic (Default)"

# download PhotoMaker checkpoint to cache
photomaker_ckpt = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model", cache_dir=cache_dir)

pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
    base_model_path, 
    torch_dtype=torch.bfloat16, 
    use_safetensors=True, 
    variant="fp16",
    cache_dir=cache_dir
    # local_files_only=True,
).to(device)

pipe.load_photomaker_adapter(
    os.path.dirname(photomaker_ckpt),
    subfolder="",
    weight_name=os.path.basename(photomaker_ckpt),
    trigger_word="img"
)
pipe.id_encoder.to(device)

pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
# pipe.set_adapters(["photomaker"], adapter_weights=[1.0])
pipe.fuse_lora()

@spaces.GPU(enable_queue=True)
def generate_image(upload_images, prompt, negative_prompt, style_name, num_steps, style_strength_ratio, num_outputs, guidance_scale, seed, progress=gr.Progress(track_tqdm=True)):
    
    # check the trigger word
    image_token_id = pipe.tokenizer.convert_tokens_to_ids(pipe.trigger_word)
    prompt = "img of " + prompt
    input_ids = pipe.tokenizer.encode(prompt)
    if image_token_id not in input_ids:
        raise gr.Error(f"Cannot find the trigger word '{pipe.trigger_word}' in text prompt! Please refer to step 2️⃣")

    if input_ids.count(image_token_id) > 1:
        raise gr.Error(f"Cannot use multiple trigger words '{pipe.trigger_word}' in text prompt!")

    # apply the style template
    prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)

    if upload_images is None:
        raise gr.Error(f"Cannot find any input face image! Please refer to step 1️⃣")

    input_id_images = []
    for img in upload_images:
        input_id_images.append(load_image(img))
    
    generator = torch.Generator(device=device).manual_seed(seed)

    print("\nStart inference...")
    print(f"[Debug] Prompt: {prompt}")
    start_merge_step = int(float(style_strength_ratio) / 100 * num_steps)
    if start_merge_step > 30:
        start_merge_step = 30
    print(start_merge_step)
    print("\n\n")
    images = pipe(
        prompt=prompt,
        input_id_images=input_id_images,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_outputs,
        num_inference_steps=num_steps,
        start_merge_step=start_merge_step,
        generator=generator,
        guidance_scale=guidance_scale,
    ).images
    return images, gr.update(visible=True)

def swap_to_gallery(images):
    return gr.update(value=images, visible=True), gr.update(visible=True), gr.update(visible=False)

def upload_example_to_gallery(images, prompt, style, negative_prompt):
    return gr.update(value=images, visible=True), gr.update(visible=True), gr.update(visible=False)

def remove_back_to_files():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
    
def remove_tips():
    return gr.update(visible=False)

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def apply_style(style_name: str, positive: str, negative: str = "") -> tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive), n + ' ' + negative

def get_image_path_list(folder_name):
    image_basename_list = os.listdir(folder_name)
    image_path_list = sorted([os.path.join(folder_name, basename) for basename in image_basename_list])
    return image_path_list

### Description and style
logo = r"""
"""
title = r"""
<h1 align="center">Nat PhotoMaker: Img to Text</h1>
"""

description = r"""
❗️❗️❗️[<b>Important</b>] Passo a passo:<br>
1️⃣  Faça upload das imagens de alguém que você deseja personalizar. Uma imagem é suficiente, mas quanto mais, melhor. O rosto na imagem carregada deve <b>ocupar a maior parte da imagem</b>.<br>
2️⃣ Insira um prompt de texto,  em inglês, explicando como a foto deve ser feita. Exemplo: a powerful world of warcraft fire mage, full-body.<br>
3️⃣ Escolha o seu modelo de estilo preferido.<br>
4️⃣ Clique no botão <b>Enviar</b> para começar a personalização.
"""

article = r"""
TencentARC/PhotoMaker: https://huggingface.co/TencentARC/PhotoMaker
"""

tips = r"""
"""

css = '''
.gradio-container {width: 85% !important}
'''
with gr.Blocks(css=css) as demo:
    gr.Markdown(logo)
    gr.Markdown(title)
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            files = gr.Files(
                        label="Adicione uma ou mais fotos do seu rosto",
                        file_types=["image"]
                    )
            uploaded_files = gr.Gallery(label="Suas fotos", visible=False, columns=5, rows=1, height=200)
            with gr.Column(visible=False) as clear_button:
                remove_and_reupload = gr.ClearButton(value="Remover e adicionar novas fotos", components=files, size="sm")
            prompt = gr.Textbox(label="Prompt",
                       info="Experimente algo como 'a powerful World of Warcraft warrior...'",
                       placeholder="A fire mage in the woods with a powerful spell...")
            style = gr.Dropdown(label="Templates", choices=STYLE_NAMES, value=DEFAULT_STYLE_NAME)
            submit = gr.Button("Enviar")

            with gr.Accordion(open=False, label="Opções Avançadas"):
                negative_prompt = gr.Textbox(
                    label="Negative Prompt", 
                    placeholder="low quality",
                    value="nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
                )
                num_steps = gr.Slider( 
                    label="Number of sample steps",
                    minimum=20,
                    maximum=100,
                    step=1,
                    value=50,
                )
                style_strength_ratio = gr.Slider(
                    label="Style strength (%)",
                    minimum=15,
                    maximum=50,
                    step=1,
                    value=20,
                )
                num_outputs = gr.Slider(
                    label="Number of output images",
                    minimum=1,
                    maximum=2,
                    step=1,
                    value=2,
                )
                guidance_scale = gr.Slider(
                    label="Guidance scale",
                    minimum=0.1,
                    maximum=10.0,
                    step=0.1,
                    value=5,
                )
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=0,
                )
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
        with gr.Column():
            gallery = gr.Gallery(label="Imagens geradas")
            usage_tips = gr.Markdown(label="Dicas de uso do PhotoMaker", value=tips ,visible=False)

        files.upload(fn=swap_to_gallery, inputs=files, outputs=[uploaded_files, clear_button, files])
        remove_and_reupload.click(fn=remove_back_to_files, outputs=[uploaded_files, clear_button, files])

        submit.click(
            fn=remove_tips,
            outputs=usage_tips,            
        ).then(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
            api_name=False,
        ).then(
            fn=generate_image,
            inputs=[files, prompt, negative_prompt, style, num_steps, style_strength_ratio, num_outputs, guidance_scale, seed],
            outputs=[gallery, usage_tips]
        )

    # gr.Examples(
    #     examples=get_example(),
    #     inputs=[files, prompt, style, negative_prompt],
    #     run_on_click=True,
    #     fn=upload_example_to_gallery,
    #     outputs=[uploaded_files, clear_button, files],
    # )
    
    gr.Markdown(article)
    
demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
