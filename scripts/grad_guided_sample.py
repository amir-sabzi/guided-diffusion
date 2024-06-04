"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

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


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")

    
     
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
            scores = th.autograd.grad(score_fn, x_t, retain_graph=True)[0] * args.classifier_scale
            return scores
             
    def model_fn(x, t, y=None, x_0=None):
        return model(x, t, y, x_0)

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
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
        
        
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
        )

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, classes)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)

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
