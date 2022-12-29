from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    DPMSolverMultistepScheduler,
)
import gradio as gr
import torch
from PIL import Image
import utils
import datetime
import time
import psutil
import os
from share_btn import community_icon_html, loading_icon_html, share_js


start_time = time.time()



text_gen = gr.Interface.load(name="spaces/Omnibus/MagicPrompt-Stable-Diffusion")
#stable_diffusion = gr.Blocks.load(name="spaces/runwayml/stable-diffusion-v1-5")





def get_prompts(prompt_text):
    return text_gen(prompt_text)


css = """
.animate-spin {
    animation: spin 1s linear infinite;
}
@keyframes spin {
    from {
        transform: rotate(0deg);
    }
    to {
        transform: rotate(360deg);
    }
}
body {
background-color: #3e567d;
text-color: #ffffff;
}
h1 {
text-color: #ffffff; !important;
color: #ffffff; !important;
}
#gallery {
min-height:60rem
}
p {
color: #ffffff; !important;
}

#share-btn-container {
    display: flex; padding-left: 0.5rem !important; padding-right: 0.5rem !important; background-color: #000000; justify-content: center; align-items: center; border-radius: 9999px !important; width: 13rem;
}
#share-btn {
    all: initial; color: #ffffff;font-weight: 600; cursor:pointer; font-family: 'IBM Plex Sans', sans-serif; margin-left: 0.5rem !important; padding-top: 0.25rem !important; padding-bottom: 0.25rem !important;
}
#share-btn * {
    all: unset;
}
#share-btn-container div:nth-child(-n+2){
    width: auto !important;
    min-height: 0px !important;
}
#share-btn-container .wrap {
    display: none !important;
}
a {text-decoration-line: underline;}
"""


class Model:
    def __init__(self, name, path="", prefix=""):
        self.name = name
        self.path = path
        self.prefix = prefix
        self.pipe_t2i = None
        self.pipe_i2i = None


models = [
    Model("Arcane", "nitrosocke/Arcane-Diffusion", "arcane style "),
    Model(
        "Dreamlike Diffusion 1.0",
        "dreamlike-art/dreamlike-diffusion-1.0",
        "dreamlikeart ",
    ),
    Model("Archer", "nitrosocke/archer-diffusion", "archer style "),
    Model("Anything V3", "Linaqruf/anything-v3.0", ""),
    Model("Modern Disney", "nitrosocke/mo-di-diffusion", "modern disney style "),
    Model(
        "Classic Disney", "nitrosocke/classic-anim-diffusion", "classic disney style "
    ),
    Model("Loving Vincent (Van Gogh)", "dallinmackay/Van-Gogh-diffusion", "lvngvncnt "),
    Model("Wavyfusion", "wavymulder/wavyfusion", "wa-vy style "),
    Model("Analog Diffusion", "wavymulder/Analog-Diffusion", "analog style "),
    Model(
        "Redshift renderer (Cinema4D)",
        "nitrosocke/redshift-diffusion",
        "redshift style ",
    ),
    Model(
        "Midjourney v4 style", "prompthero/midjourney-v4-diffusion", "mdjrny-v4 style "
    ),
    Model("Waifu", "hakurei/waifu-diffusion"),
    Model(
        "Cyberpunk Anime",
        "DGSpitzer/Cyberpunk-Anime-Diffusion",
        "dgs illustration style ",
    ),
    Model("Elden Ring", "nitrosocke/elden-ring-diffusion", "elden ring style "),
    Model("TrinArt v2", "naclbit/trinart_stable_diffusion_v2"),
    Model("Spider-Verse", "nitrosocke/spider-verse-diffusion", "spiderverse style "),
    Model("Balloon Art", "Fictiverse/Stable_Diffusion_BalloonArt_Model", "BalloonArt "),
    Model("Tron Legacy", "dallinmackay/Tron-Legacy-diffusion", "trnlgcy "),
    Model("Pokémon", "lambdalabs/sd-pokemon-diffusers"),
    Model("Pony Diffusion", "AstraliteHeart/pony-diffusion"),
    Model("Robo Diffusion", "nousr/robo-diffusion"),
]

custom_model = None




last_mode = "txt2img"
current_model = models[0]
current_model_path = current_model.path



pipe = StableDiffusionPipeline.from_pretrained(
  current_model.path,
  torch_dtype=torch.get_default_dtype(),
  scheduler=DPMSolverMultistepScheduler.from_pretrained(current_model.path, subfolder="scheduler")
  )
    
if torch.cuda.is_available():
  pipe = pipe.to("cuda")
  pipe.enable_xformers_memory_efficient_attention()

device = "GPU 🔥" if torch.cuda.is_available() else "CPU 🥶"

def error_str(error, title="Error"):
    return f"""#### {title}
            {error}"""  if error else ""

def custom_model_changed(path):
  models[0].path = path
  global current_model
  current_model = models[0]

def on_model_change(model_name):
  
  prefix = "Enter prompt. \"" + next((m.prefix for m in models if m.name == model_name), None) + "\" is prefixed automatically" if model_name != models[0].name else "Don't forget to use the custom model prefix in the prompt!"

  return gr.update(visible = model_name == models[0].name), gr.update(placeholder=prefix)

def inference(model_name, prompt, guidance, steps, n_images=1, width=512, height=512, seed=0, img=None, strength=0.5, neg_prompt=""):

  print(psutil.virtual_memory()) # print memory usage

  global current_model
  for model in models:
    if model.name == model_name:
      current_model = model
      model_path = current_model.path

  generator = torch.Generator('cuda').manual_seed(seed) if seed != 0 else None

  try:
    if img is not None:
      return img_to_img(model_path, prompt, n_images, neg_prompt, img, strength, guidance, steps, width, height, generator), None
    else:
      return txt_to_img(model_path, prompt, n_images, neg_prompt, guidance, steps, width, height, generator), None
  except Exception as e:
    return None, error_str(e)

def txt_to_img(model_path, prompt, n_images, neg_prompt, guidance, steps, width, height, generator):

    print(f"{datetime.datetime.now()} txt_to_img, model: {current_model.name}")

    global last_mode
    global pipe
    global current_model_path
    if model_path != current_model_path or last_mode != "txt2img":
        current_model_path = model_path

        if current_model == custom_model:
          pipe = StableDiffusionPipeline.from_pretrained(
              current_model_path,
              torch_dtype=torch.get_default_dtype(),
              scheduler=DPMSolverMultistepScheduler.from_pretrained(current_model.path, subfolder="scheduler"),
              safety_checker=lambda images, clip_input: (images, False)
              )
        else:
          pipe = StableDiffusionPipeline.from_pretrained(
              current_model_path,
              torch_dtype=torch.get_default_dtype(),
              scheduler=DPMSolverMultistepScheduler.from_pretrained(current_model.path, subfolder="scheduler")
              )
          # pipe = pipe.to("cpu")
          # pipe = current_model.pipe_t2i

        if torch.cuda.is_available():
          pipe = pipe.to("cuda")
          pipe.enable_xformers_memory_efficient_attention()
        last_mode = "txt2img"

    prompt = current_model.prefix + prompt  
    result = pipe(
      prompt,
      negative_prompt = neg_prompt,
      num_images_per_prompt=n_images,
      num_inference_steps = int(steps),
      guidance_scale = guidance,
      width = width,
      height = height,
      generator = generator)
    
    return replace_nsfw_images(result)

def img_to_img(model_path, prompt, n_images, neg_prompt, img, strength, guidance, steps, width, height, generator):

    print(f"{datetime.datetime.now()} img_to_img, model: {model_path}")

    global last_mode
    global pipe
    global current_model_path
    if model_path != current_model_path or last_mode != "img2img":
        current_model_path = model_path

        if current_model == custom_model:
          pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
              current_model_path,
              torch_dtype=torch.get_default_dtype(),
              scheduler=DPMSolverMultistepScheduler.from_pretrained(current_model.path, subfolder="scheduler"),
              safety_checker=lambda images, clip_input: (images, False)
              )
        else:
          pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
              current_model_path,
              torch_dtype=torch.get_default_dtype(),
              scheduler=DPMSolverMultistepScheduler.from_pretrained(current_model.path, subfolder="scheduler")
              )
          #pipe = pipe.to("cpu")
          # pipe = current_model.pipe_i2i
        
        if torch.cuda.is_available():
          pipe = pipe.to("cuda")
          pipe.enable_xformers_memory_efficient_attention()
        last_mode = "img2img"

    prompt = current_model.prefix + prompt
    ratio = min(height / img.height, width / img.width)
    img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    result = pipe(
        prompt,
        negative_prompt = neg_prompt,
        num_images_per_prompt=n_images,
        image = img,
        num_inference_steps = int(steps),
        strength = strength,
        guidance_scale = guidance,
        # width = width,
        # height = height,
        generator = generator)
        
    return replace_nsfw_images(result)

def replace_nsfw_images(results):

    
      
    for i in range(len(results.images)):
      if results.nsfw_content_detected[i]:
        results.images[i] = Image.open("nsfw.png")
    return results.images

#css = """.finetuned-diffusion-div div{display:inline-flex;align-items:center;gap:.8rem;font-size:1.75rem}.finetuned-diffusion-div div h1{font-weight:900;margin-bottom:7px}.finetuned-diffusion-div p{margin-bottom:10px;font-size:94%}a{text-decoration:underline}.tabs{margin-top:0;margin-bottom:0}#gallery{min-height:60rem}
#css = """.finetuned-diffusion-div div{display:inline-flex;align-items:center;gap:.8rem;font-size:1.75rem}.finetuned-diffusion-div div h1{font-weight:900;margin-bottom:7px}.finetuned-diffusion-div p{margin-bottom:10px;font-size:94%}a{text-decoration:underline}.tabs{margin-top:0;margin-bottom:0}#gallery{min-height:60rem}

#"""



with gr.Blocks(css=css) as demo:
    gr.HTML(
        """<div style="text-align: center; max-width: 700px; margin: 0 auto;">
            <div
            style="
                display: inline-flex;
                align-items: center;
                gap: 0.8rem;
                font-size: 1.75rem;
            "
            >
            <h1 style="font-weight: 900; text-color: #ffffff; margin-bottom: 7px; margin-top: 5px;">
                Finetuned Magic
            </h1>
            </div>
            <p style="margin-bottom: 10px; font-size: 94%">
            This Space prettifies your prompt using <a href="https://huggingface.co/spaces/Gustavosta/MagicPrompt-Stable-Diffusion" target="_blank">MagicPrompt</a>
            and then runs it through <a href="https://huggingface.co/spaces/anzorq/finetuned_diffusion" target="_blank">Finetuned Diffusion</a> to create aesthetically pleasing images. Simply enter a few concepts and let it improve your prompt. You can then diffuse the prompt.
            </p>
             <p style="margin-bottom: 10px; font-size: 94%">This can take longer than 15 minutes to Render out one image using Redshift Render on CPU</p>
            <p style="margin-bottom: 10px; font-size: 94%">Don't Give Up!</p>
        </div>"""
    )

    with gr.Row():
        with gr.Column(scale=20):
            input_text = gr.Textbox(
                label="Short text prompt", lines=4, elem_id="input-text"
            )
            with gr.Row():
                see_prompts = gr.Button("Feed in your text!")

        
            prompt = gr.Textbox(
                label="Prettified text prompt", lines=4, elem_id="translated"
            )
            with gr.Row():
               

                generate = gr.Button(value="Generate").style(
                    rounded=(False, True, True, False)                
                )


            with gr.Tab("Options"):
                with gr.Group():
                    neg_prompt = gr.Textbox(
                        label="Negative prompt",
                        placeholder="What to exclude from the image",
                    )

                    n_images = gr.Slider(
                        label="Images", value=1, minimum=1, maximum=4, step=1
                    )

                    with gr.Row():
                        guidance = gr.Slider(
                            label="Guidance scale", value=7.5, maximum=15
                        )
                        steps = gr.Slider(
                            label="Steps", value=25, minimum=2, maximum=75, step=1
                        )

                    with gr.Row():
                        width = gr.Slider(
                            label="Width",
                            value=512,
                            minimum=64,
                            maximum=1024,
                            step=8,
                        )
                        height = gr.Slider(
                            label="Height",
                            value=512,
                            minimum=64,
                            maximum=1024,
                            step=8,
                        )

                    seed = gr.Slider(
                        0, 2147483647, label="Seed (0 = random)", value=0, step=1
                    )

            with gr.Tab("Image to image"):
                with gr.Group():
                    image = gr.Image(
                        label="Image", height=256, tool="editor", type="pil"
                    )
                    strength = gr.Slider(
                        label="Transformation strength",
                        minimum=0,
                        maximum=1,
                        step=0.01,
                        value=0.5,
                    )
            with gr.Tab("New Tab"):
                with gr.Group():
                    image = gr.Image(
                        label="Image", height=256, tool="editor", type="pil"
                    )
                    strength = gr.Slider(
                        label="Transformation strength",
                        minimum=0,
                        maximum=1,
                        step=0.01,
                        value=0.5,
                    )
        with gr.Column(scale=80):
          with gr.Group():
              model_name = gr.Dropdown(label="Model", choices=[m.name for m in models], value=current_model.name)
              with gr.Box(visible=False) as custom_model_group:
                custom_model_path = gr.Textbox(label="Custom model path", placeholder="Path to model, e.g. nitrosocke/Arcane-Diffusion", interactive=True)
                gr.HTML("<div><font size='2'>Custom models have to be downloaded first, so give it some time.</font></div>")
              
              


              # image_out = gr.Image(height=512)
              gallery = gr.Gallery(label="Generated images", show_label=False, elem_id="gallery").style(grid=[2], height="auto")

          error_output = gr.Markdown()

  
    with gr.Row():

        with gr.Group(elem_id="share-btn-container"):
            community_icon = gr.HTML(community_icon_html, visible=False)
            loading_icon = gr.HTML(loading_icon_html, visible=False)
            share_button = gr.Button(
                "Share to community", elem_id="share-btn", visible=False
            )
    gr.HTML("""
            <div style="border-top: 1px solid #303030;">
              <br>
              <p><img src="https://visitor-badge.glitch.me/badge?page_id=omnibus.finetuned-magic_dev" alt="visitors"></p>
             
              <p>Models by <a href="https://huggingface.co/Gustavosta">@Gustavosta</a>, <a href="https://twitter.com/haruu1367">@haruu1367</a>, <a href="https://twitter.com/DGSpitzer">@Helixngc7293</a>, <a href="https://twitter.com/dal_mack">@dal_mack</a>, <a href="https://twitter.com/prompthero">@prompthero</a> and others.</p>
            </div>
            """)
    gr.HTML("""
            <div style="margin-top:50px; border-top: 1px solid #303030;">
              <br>
              <a href="https://www.buymeacoffee.com/Omnibus" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-blue.png" alt="Buy Me A Coffee" style="height: 45px !important;width: 162px !important;" ></a><br><br>
            </div>
            """)



    
    see_prompts.click(get_prompts, inputs=[input_text], outputs=[prompt])
    
    inputs = [model_name, prompt, guidance, steps, n_images, width, height, seed, image, strength, neg_prompt]
    outputs = [gallery, error_output]
    
    prompt.submit(inference, inputs=inputs, outputs=outputs)
    generate.click(inference, inputs=inputs, outputs=outputs)
   
    share_button.click(None, [], [], _js=share_js)
    
demo.queue(concurrency_count=1)
demo.launch(debug=True)