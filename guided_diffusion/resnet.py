

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv












def main():
    
    resNet = tv.models.resnet50(pretrained=True)
    print(resNet)
    







'''
python scripts/grad_guided_sample.py \
  --attention_resolutions 32,16,8 \
  --class_cond True \
  --diffusion_steps 1000 \
  --image_size 256 \
  --learn_sigma True \
  --noise_schedule linear \
  --num_channels 256 \
  --batch_size 2 \
  --num_head_channels 64 \
  --num_res_blocks 2 \
  --resblock_updown True \
  --use_fp16 True \
  --use_scale_shift_norm True \
  --classifier_scales "0"\
  --num_iters 1 \
  --classifier_path models/256x256_classifier.pt \
  --model_path models/256x256_diffusion.pt \
  --data_dir /home/amirsabzi/data/imagenet/ILSVRC/Data/CLS-LOC/partioned_train/dir1 \
  --log_dir debug
  
  
python scripts/grad_guided_sample.py \
  --attention_resolutions 32,16,8 \
  --class_cond True \
  --diffusion_steps 1000 \
  --image_size 256 \
  --learn_sigma True \
  --noise_schedule linear \
  --num_channels 256 \
  --batch_size 2 \
  --num_head_channels 64 \
  --num_res_blocks 2 \
  --resblock_updown True \
  --use_fp16 True \
  --use_scale_shift_norm True \
  --classifier_scales "0"\
  --num_iters 1 \
  --target_model_path checkpoints/checkpoint_10000.pth \
  --model_path models/256x256_diffusion.pt \
  --data_dir /home/amirsabzi/data/imagenet/ILSVRC/Data/CLS-LOC/partioned_train/dir1 \
  --log_dir debug \
  --num_classes 100 \
  --use_ddim True 
   
  
  
  
  
'''



if __name__ == '__main__':
    main()