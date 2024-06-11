"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
import matplotlib.pyplot as plt
import datetime
import os
import imageio


from guided_diffusion import dist_util, logger
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



def save_images(images, filename, plot_dir):
    """
    Saves a batch of images to the specified directory.
    
    Args:
    images (Tensor): Batch of images.
    filename (str): Filename for the saved plot.
    plot_dir (str): Directory to save the plots.
    """
    images = ((images + 1) * 127.5).clamp(0, 255).to(th.uint8)
    images = images.permute(0, 2, 3, 1)
    images = images.contiguous().cpu().numpy()
    # Create a directory for plots if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot and save images
    num_images = len(images)
    fig, axs = plt.subplots(1, num_images, figsize=(num_images * 2, 2))
    if num_images == 1:
        axs = [axs]  # Make it iterable if there's only one subplot
    for i, img in enumerate(images):
        axs[i].imshow(img)
        axs[i].axis('off')
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close(fig)
    
def create_gif(images_array, filename, gif_dir, duration=1):
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
    images_array = [((image.squeeze(0) + 1) * 127.5).clamp(0, 255).to(th.uint8).permute(1, 2, 0).contiguous().cpu().numpy() for image in images_array]
    # Save frames as GIF

    gif_path = os.path.join(gif_dir, filename)
    imageio.mimsave(gif_path, images_array, duration=duration)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()

    log_dir = dir = os.path.join(
            "logs",
            datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"),
        ) 
    
    os.makedirs(log_dir, exist_ok=True) 
    logger.configure(dir=log_dir)

    logger.log("creating model and diffusion...")

    logger.arg_logger(args) 
     
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()

    model.eval()

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    
    logger.log("creating data loader...")
    data = load_data(data_dir=args.data_dir,batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=True,
        random_crop=False,
    )
    data_iter = iter(data)

    def get_grads(x, y): 
        x.requires_grad = True
        t = th.zeros(x.size(0), device=x.device)
        logits = classifier(x, t)
        loss = F.cross_entropy(logits, y)


    
        params = list(classifier.parameters())
        grad_params = th.autograd.grad(loss, params, create_graph=True)
        grad_params_tensor = th.cat([grad_param.view(-1) for grad_param in grad_params]) 
        
         
        # grad_params_l2 = th.norm(grad_params_tensor, p=2)        
        
        # grad_l2_norm = th.autograd.grad(grad_params_l2, x, retain_graph=True) 
        
        return grad_params_tensor

    def cond_fn(x_t, t, y=None, x_0=None): 
        # return 0
        with th.enable_grad():
            # Ensure x_t and x_0 are not detached and have requires_grad=True
            x_t = x_t.clone().detach().requires_grad_(True)
            x_0 = x_0.clone().detach().requires_grad_(True)
            
            # # Debug: Print to ensure tensors have requires_grad=True
            # print("x_t.requires_grad:", x_t.requires_grad)
            # print("x_0.requires_grad:", x_0.requires_grad)

            grads_x_t = get_grads(x_t, y)
            grads_x_0 = get_grads(x_0, y)
            
            # Score function is the \|grads_x_t - grads_x_0\|_2^2
            score_fn = th.norm(grads_x_t - grads_x_0, p=2)**2
            logger.logkv("score_fn", score_fn.item())
            # logger.dumpkvs()
            # print("score_fn:", score_fn)
            scores = th.autograd.grad(score_fn, x_t, retain_graph=True)[0] * args.classifier_scale
            return scores
             
    def model_fn(x, t, y=None, x_0=None):
        return model(x, t, y, x_0)

    logger.log("sampling...")
    plot_dir = os.path.join(log_dir, "plots")   
    for i in range(args.num_samples):
        model_kwargs = {}
        classes = th.randint(
            low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
        )
        x, extra = next(data_iter)
        y = extra["y"]

        

        # put data and labels on the same device as the model
        model_kwargs["x_0"], model_kwargs["y"] = x.to(dist_util.dev()), y.to(dist_util.dev()) 
           
        
         

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        
        
        sample, diffusion_step = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
        )

        save_images(x, f"original_{i}.png", plot_dir)

        save_images(sample, f"sample_{i}.png", plot_dir) 

        create_gif(diffusion_step, f"diffusion_{i}.gif", plot_dir)

        logger.log(f"created {i+1} samples")


    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        data_dir="",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
