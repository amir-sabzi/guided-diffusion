"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
from unittest import result

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
    target_model_defaults,
    create_model_and_diffusion,
    create_target_model,
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

def plot_score(data_dir, num_iters, diffusion_steps):
    csv_file = os.path.join(data_dir, "progress.csv")
    # read the csv file at data dir
    with open(csv_file, mode='r') as file:
        df = pd.read_csv(file)

    # get unique scales
    scales = df['scale'].unique()
    for scale in scales:
        # get the data for each scale
        df_scale = df[df['scale'] == scale].reset_index()

        # round the scale to 2 decimal places 
        rounded_coefficient, exponent = round_to_nearest_i_times_10x(scale)
        
       
        label = f"scale: {rounded_coefficient}e{exponent}" 
        score_fn = np.zeros(diffusion_steps) 
        for i in range(num_iters):
            if df_scale['score_fn'][i*diffusion_steps:(i+1)*diffusion_steps].isna().any():
                label = f"scale: {rounded_coefficient}e{exponent}, FAILED"

            score_fn += df_scale['score_fn'][i*diffusion_steps:(i+1)*diffusion_steps].to_numpy()
        
        score_fn = score_fn / num_iters   
        
        # plot the score_fn vs index for each scale
        plt.plot(range(diffusion_steps), score_fn, label=label)  
        
    plt.xlabel('diffusion step')
    plt.ylabel('Gradient l2 difference')
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.legend()
    plt.savefig(os.path.join(data_dir, "plots", "score_fn.png"))
    plt.close() 





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
        data = results[keys[i]]
        data = ((data + 1) * 127.5).clamp(0, 255).to(th.uint8)
        data = data.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        for j in range(num_cols):
            axs[i][j].imshow(data[j])
            axs[i][j].axis('off')
    
    
     
    for row in range(num_rows):
        if keys[row] == "x":
            axs[row, 0].text(-40, 128, 'data', rotation=90, fontsize=16, va='center')
        else:
            scale = keys[row].item()
            c, e = round_to_nearest_i_times_10x(scale) 
            axs[row, 0].text(-20, 128, f's={c}e{e}', rotation=90, fontsize=16, va='center') 
    
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

def get_finename(scale, iteration, prefix=""):
    if scale == 0:
        return f"{prefix}_iter={iteration}_s=0.pdf"
    # map tensor to float
    scale = scale.item()
    rounded_coefficient, exponent = round_to_nearest_i_times_10x(scale)
    return f"{prefix}_iter={iteration}_s={rounded_coefficient}e{exponent}.pdf" 

def main():


    args = create_argparser().parse_args()

    # log_dir = "/home/amirsabzi/workspace/guided-diffusion/scales/gdg-2024-06-15-11-58-47-193957/"

    # plot_score(log_dir, args.num_iters, args.diffusion_steps)
    
    # return
    # dist_util.setup_dist()
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

    logger.log("creating model and diffusion...")

    logger.arg_logger(args) 
     
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

    logger.log("loading target model...")
    target_model = create_target_model(**args_to_dict(args, target_model_defaults().keys()))

    
    checkpoint = th.load(args.target_model_path, map_location="cpu") 
     
    target_model.load_state_dict(checkpoint['model_state_dict']) 

    target_model.to(sg_util.dev())
    target_model.eval()

    
    logger.log("creating data loader...")
    data = load_data(data_dir=args.data_dir,batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=True,
        random_crop=False,
    )
    # data_iter = iter(data)

    def get_grads(x, y): 
        x.requires_grad = True
        # t = th.zeros(x.size(0), device=x.device)
        logits = target_model(x)
        loss = F.cross_entropy(logits, y)


    
        params = list(target_model.parameters())
        grad_params = th.autograd.grad(loss, params, create_graph=True)
        grad_params_tensor = th.cat([grad_param.view(-1) for grad_param in grad_params]) 
        
         
        # grad_params_l2 = th.norm(grad_params_tensor, p=2)        
        
        # grad_l2_norm = th.autograd.grad(grad_params_l2, x, retain_graph=True) 
        
        return grad_params_tensor

    def cond_fn(x_t, t, y=None, x_0=None, s=None): 
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
            logger.logkv("scale", s.item()) 
            logger.dumpkvs()
            # logger.dumpkvs()
            # print("score_fn:", score_fn)
            scores = th.autograd.grad(score_fn, x_t, retain_graph=True)[0] 
            logger.logkv("score norm", th.norm(scores, p=2).item())
            scores = scores * s
            return scores
             
    def model_fn(x, t, y=None, x_0=None, s=None):
        return model(x, t, y, x_0, s)

    logger.log("sampling...")
    plot_dir = os.path.join(log_dir, "plots")   
    classifier_scales = args.classifier_scales 
    classifier_scales = th.tensor([float(x) for x in classifier_scales.split(",")]) if classifier_scales else th.tensor([0.0])
    for i in range(args.num_iters):
        model_kwargs = {}

        x, extra = next(data)
        y = extra["y"]

        results = {}
        results["x"] = x
        for scale in classifier_scales: 
        
        
            # put data and labels on the same device as the model
            model_kwargs["x_0"], model_kwargs["y"], model_kwargs["s"] = x.to(sg_util.dev()), y.to(sg_util.dev()), scale.to(sg_util.dev()) 
            
            
            

            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )

            sample, diffusion_step = sample_fn(
                model_fn,
                (args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
                device=sg_util.dev(),
            )

            
            results[scale] = sample
            
        save_images(results,
                    num_rows = len(classifier_scales) + 1,
                    num_cols = args.batch_size,
                    filename = get_finename(0, i, "data"),
                    plot_dir = plot_dir)


    
    
    
    
    logger.log("sampling complete")
    plot_score(log_dir, args.num_iters, args.diffusion_steps)

def create_argparser():
    defaults = dict(
        data_dir="",
        log_dir="",
        clip_denoised=True,
        num_iters=100,
        batch_size=16,
        use_ddim=False,
        model_path="",
        target_model_path="",
        classifier_scales="",
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(target_model_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
