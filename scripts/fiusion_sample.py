'''
Sample intermidate steps of noise diffusion process. 
process towardss true Gaussian noise.
'''

import argparse


import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
import matplotlib.pyplot as plt
import datetime
import os
import pandas as pd
import imageio


from guided_diffusion import sg_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    create_target_model,
    add_dict_to_argparser,
    args_to_dict,
)



def save_images(samples, filename, plot_dir):
    """
    Saves a batch of sampled images to the specified directory.

    Args:
    samples (Tensor): Batch of sampled images.
    filename (str): Filename for the saved plot.
    plot_dir (str): Directory to save the plots.
    """
    samples = ((samples + 1) * 127.5).clamp(0, 255).to(th.uint8)

    samples = samples.permute(0, 2, 3, 1).contiguous().cpu().numpy()

    # Create a directory for plots if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)

    # Plot and save images
    num_images = len(samples)
    fig, axs = plt.subplots(1, num_images, figsize=(num_images * 2, 2))

    if num_images == 1:
        axs = [axs]  # Make it iterable if there's only one subplot

    for i in range(num_images):
        axs[i].imshow(samples[i])
        axs[i].axis('off')

    plt.savefig(os.path.join(plot_dir, filename))
    plt.close(fig)
 

   
def create_gif(images_array, filename, gif_dir, duration=0.4):
    """
    Create a GIF from a list of images.
    
    Args:
    images (Tensor): List of images (Tensor).
    filename (str): Filename for the saved GIF.
    gif_dir (str): Directory to save the GIF.
    duration (float): Duration (in seconds) of each frame in the GIF.
    """
    # Create a directory for GIFs if it doesn't exist
    os.makedirs(gif_dir, exist_ok=True)
    
    # Transform each image in images_array to numpy arrays with shape (64, 64, 3)
    images_array = [((image + 1) * 127.5).clamp(0, 255).to(th.uint8).permute(1, 2, 0).contiguous().cpu().numpy() for image in images_array]
    # Save frames as GIF

    
    gif_path = os.path.join(gif_dir, filename)
    imageio.mimsave(gif_path, images_array, duration=duration)

def main():
    args = create_argparser().parse_args()

    if args.log_dir: 
        log_dir_root = args.log_dir
    else: 
        log_dir_root = "logs";
    
    log_dir = os.path.join(
            log_dir_root,
            datetime.datetime.now().strftime("gdg-%Y-%m-%d-%H-%M-%S-%f"),
        ) 
    os.makedirs(log_dir, exist_ok=True) 
    logger.configure(dir=log_dir)

    logger.log("creating diffusion...")

    logger.arg_logger(args) 
     
    _ , diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    data = load_data(data_dir=args.data_dir,batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=True,
        random_crop=False,
    )
 

    
    for i in range(args.num_samples):
        
        x, _ = next(data)
        x = x.to(sg_util.dev())
        
        
          
        sample_fn = diffusion.q_sample 
        
        samples = []
        logger.log(f"creating sample {i}...")
        save_images(x, f"sample_{i}_step_0.png", log_dir)
        for step in range(args.diffusion_steps):
            step = th.tensor(step).to(sg_util.dev())
            sample = sample_fn(x, step)
            samples.append(sample)
            if not step % args.sample_save_freq:
                save_images(sample, f"sample_{i}_step_{step+1}.png", log_dir)
            
        save_images(sample, f"sample_{i}_step_{step+1}.png", log_dir)

        # create_gif(samples, f"sample_{i}.gif", log_dir)        
        
        
        
def create_argparser():
    defaults = dict(
        data_dir="",
        log_dir="",
        model_path="",
        batch_size=1,
        num_samples=10,
        sample_save_freq=100,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
