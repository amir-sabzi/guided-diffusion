"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os
import datetime
import numpy as np
import torch as th
import matplotlib.pyplot as plt
import torch.distributed as dist
import torch.nn.functional as F

from guided_diffusion import sg_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)



def round_to_nearest_i_times_10x(scale):
    if scale == 0:
        return 0, 0
    
    exponent = int(np.floor(np.log10(scale)))
    coefficient = scale / (10 ** exponent)
    # print("coefficient:", coefficient)
    rounded_coefficient = round(coefficient)
    # print("rounded_coefficient:", rounded_coefficient)
    return rounded_coefficient, exponent


def get_finename(scale, iteration, prefix=""):
    if scale == 0:
        return f"{prefix}_iter={iteration}_s=0.pdf"
    # map tensor to float
    scale = scale.item()
    rounded_coefficient, exponent = round_to_nearest_i_times_10x(scale)
    return f"{prefix}_iter={iteration}_s={rounded_coefficient}e{exponent}.pdf" 


def image_transform(image):
    image = ((image + 1) * 127.5).clamp(0, 255).to(th.uint8)
    image = image.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    return image    


def save_images(results, num_rows, num_cols, filename, plot_dir):
    """
    Saves a batch of images and their corresponding samples to the specified directory.
    
    Args:
    results (dict): Dictionary containing the original images and their corresponding samples.
    num_rows (int): Number of rows in the plot.
    num_cols (int): Number of columns in the plot.
    filename (str): Filename for the saved plot.
    plot_dir (str): Directory to save the plots.    

    """


    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot and save images
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 15))
    keys = list(results.keys()) 
    for i in range(num_rows):
        data = results[keys[i]][0]
        data = image_transform(data)
        axs_flat = axs.flatten()
        for j in range(num_cols):
            axs_flat[i * num_cols + j].imshow(data[j])
            axs_flat[i * num_cols + j].axis('off')
     
    for row in range(num_rows):
        if keys[row] == "x":
            axs[row, 0].text(-40, 128, 'data', rotation=90, fontsize=16, va='center')
        else:
            scale = keys[row].item()
            c, e = round_to_nearest_i_times_10x(scale) 
            axs[row, 0].text(-20, 128, f's={c}e{e}', rotation=90, fontsize=16, va='center') 
    
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close(fig)




def save_gif(results, gif_dir, plot_samples=True, duration=0.4):
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
    
    if plot_samples:
        index, prefix = 1, "sample"
    else:
        index, prefix = 2, "clean_images"
    
     
    keys = list(results.keys()) 
    for key in keys:
        if key == "x":
            continue
        images = results[key][0]
        for i, image in enumerate(images):
            filename = get_finename(key, i, prefix)
        
        
        
        gif_path = os.path.join(gif_dir, f'{prefix}_s={key}.gif')
        
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
        log_dir_root = "classifier_logs";
     
    log_dir = os.path.join(
            log_dir_root,
            datetime.datetime.now().strftime("gdg-%Y-%m-%d-%H-%M-%S-%f"),
        ) 
    os.makedirs(log_dir, exist_ok=True) 
    logger.configure(dir=log_dir)


    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        sg_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(sg_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        sg_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(sg_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()


    logger.log("creating data loader...")
    data = load_data(data_dir=args.data_dir,batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=True,
        random_crop=False,
    )
    

    def cond_fn(x, t, y=None, s=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            logger.dumpkvs()
            return th.autograd.grad(selected.sum(), x_in)[0] * s

    def model_fn(x, t, y=None, s=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None, s=s)

    logger.log("sampling...")
    plot_dir = os.path.join(log_dir, "plots")

    classifier_scales = args.classifier_scales 
    classifier_scales = th.tensor([float(x) for x in classifier_scales.split(",")]) if classifier_scales else th.tensor([0.0])
    
    
    results = {}    
    for i in range(args.num_iters):
        model_kwargs = {}
        x, extra = next(data)
        classes = extra["y"]                    
        results["x"] = x
        for scale in classifier_scales:

            model_kwargs["y"], model_kwargs["s"] = classes.to(sg_util.dev()), scale.to(sg_util.dev())
            
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            sample, diffusion_step, clean_images = sample_fn(
                model_fn,
                (args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
                device=sg_util.dev(),
            )
            results[scale] = (sample, diffusion_step, clean_images)

        save_images(results=results,
                num_rows=len(classifier_scales),
                num_cols=args.batch_size,
                filename=get_finename(0, i, "data"),
                plot_dir= plot_dir)
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        data_dir="",
        num_iters=1,
        log_dir="",
        batch_size=4,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scales="",
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
