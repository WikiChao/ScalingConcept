import torch
from PIL import Image

from src.eunms import Model_Type, Scheduler_Type
from src.utils.enums_utils import get_pipes
from src.config import RunConfig

from main import run as invert


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_type = Model_Type.SDXL
scheduler_type = Scheduler_Type.DDIM
pipe_inversion, pipe_inference = get_pipes(model_type, scheduler_type, device=device)

input_image = Image.open("example_images/new_cat_3.jpeg").convert("RGB") # "009698.jpg", "Arknight.jpg" 
original_shape = input_image.size
input_image = input_image.resize((1024, 1024))
prompt = "cat" # 'smile' for "009698.jpg", 'anime' for "Arknight.jpg"

config = RunConfig(model_type = model_type,
                    num_inference_steps = 50,
                    num_inversion_steps = 50,
                    num_renoise_steps = 1,
                    scheduler_type = scheduler_type,
                    perform_noise_correction = False,
                    seed = 7865)

_, inv_latent, _, all_latents, other_kwargs = invert(input_image,
                                       prompt,
                                       config,
                                       pipe_inversion=pipe_inversion,
                                       pipe_inference=pipe_inference,
                                       do_reconstruction=False)

rec_image = pipe_inference(image = inv_latent,
                           prompt = "",
                           denoising_start=0.0,
                           num_inference_steps = config.num_inference_steps,
                           guidance_scale = 1.0,
                            omega=5, # omega=3 for "009698.jpg", omega=5 for "Arknight.jpg"
                            gamma=3, # gamma=3 for "009698.jpg", gamma=3 for "Arknight.jpg"
                            inv_latents=all_latents,
                            prompt_embeds_ref=other_kwargs[0],
                            added_cond_kwargs_ref=other_kwargs[1],
                            t_exit=15, # t_exit=15 for "009698.jpg", t_exit=25 for "Arknight.jpg"
                            ).images[0]

rec_image.resize(original_shape).save("new_cat_3.jpg")