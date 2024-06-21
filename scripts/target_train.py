"""
Train a classification model on ImageNet that works as our target model for gradient inversion attack.
"""


import argparse
import os
import datetime

import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from guided_diffusion import sg_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    target_model_defaults,
    args_to_dict,
    create_target_model,
)

# from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict

def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)


def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)

def log_loss_dict(losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())


def save_model(model, optimizer, step, path):
    path = os.path.join(path, "checkpoints")
    os.makedirs(path, exist_ok=True)
    checkpoint_name = os.path.join(path, f"checkpoint_{step}.pth")
    th.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_name)

  
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
    print(log_dir) 
    os.makedirs(log_dir, exist_ok=True) 
    logger.configure(dir=log_dir) 
    
    
    logger.log("loading the classifer...")
    
    target_model = create_target_model(**args_to_dict(args, target_model_defaults().keys()))
    if args.target_model_path is not None:
        target_model.load_state_dict(
        sg_util.load_state_dict(args.target_model_path, map_location="cpu")
        )

    target_model.to(sg_util.dev())
    target_model.train()
    


    logger.log("creating data loader...") 
    data = load_data(

        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=True,
    )
    
    if args.val_data_dir:
        val_data = load_data(
            data_dir=args.val_data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=True,
        )
    else:
        val_data = None 


    logger.log("creating optimizer...")
    optimizer = AdamW(target_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)    
    
    
    logger.log("creating criterion...") 
    criterion = F.cross_entropy 
    
    
 
    def forward_backward_log(data_loader, prefix="train"):
        batch, extra = next(data_loader) 
        batch, labels = batch.to(sg_util.dev()), extra["y"].to(sg_util.dev()) 
        
       
        for i, (sub_batch, sub_labels) in enumerate(
            split_microbatches(args.microbatch, batch, labels)
        ):
            
            # Forward pass 
            logits = target_model(sub_batch)
            loss = criterion(logits, sub_labels) 
            
            
            losses = {}
            losses[f"{prefix}_loss"] = loss.detach()
            losses[f"{prefix}_acc@1"] = compute_top_k(
                logits, sub_labels, k=1, reduction="none"
            )
            losses[f"{prefix}_acc@5"] = compute_top_k(
                logits, sub_labels, k=5, reduction="none"
            )
            
            log_loss_dict(losses)
            del losses

            loss = loss.mean()
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
        
            
             
    # TODO: Implement laoding from checkpoints later    
    for step in range(args.num_iters):
        forward_backward_log(data, prefix="train")
        if val_data is not None and not step % args.eval_interval:
            with th.no_grad():
                target_model.eval()
                forward_backward_log(val_data, prefix="val")
                target_model.train()
        if not step % args.log_interval:
            logger.dumpkvs()
        if not step % args.save_interval:
            save_model(target_model, optimizer, step, log_dir)             

def create_argparser():
    defaults = dict(
        data_dir="",
        val_data_dir="",
        log_dir="",
        noised=True,
        num_iters=150000,
        lr=3e-4,
        weight_decay=0.0,
        microbatch=-1,
        anneal_lr=False,
        batch_size=4,
        image_size=256,
        log_interval=10,
        eval_interval=5,
        save_interval=1000,
        target_model_path=None,
    )
    defaults.update(target_model_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()



'''
python scripts/target_train.py \
  --model_name resnet50 \
  --data_dir /home/amirsabzi/data/imagenet/ILSVRC/Data/CLS-LOC/partioned_train/dir1 \
  --log_dir debug \
  --pretrained False \
  --progress False \
  --num_classes 100 \
  --batch_size 64 \
  --num_iters 5000 \
  --lr 0.01
'''