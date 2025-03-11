#!/usr/bin/env python
# -*- coding: utf-8 -*-
from backbones.vit import PatchEmbed, Attention
import torch
from collections import OrderedDict


def test_patch_embed(input_shape=[1, 3, 112, 112],
                     weight_path="./output/test_patch_embed.pth"):
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    patch_embed = PatchEmbed(img_size=input_shape[3]).to(device)
    input_tensor = torch.randn(input_shape, dtype=torch.float32, device=device)
    state_dict = OrderedDict()
    state_dict["net_state"] = patch_embed.state_dict()
    state_dict["input_tensor"] = input_tensor
    output_tensor = patch_embed(input_tensor)
    state_dict["output_tensor"] = output_tensor
    torch.save(state_dict, weight_path)


def test_attention(input_shape=[2, 144, 512],
                   weight_path="./output/test_attention.pth"):
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    atten = Attention(dim=input_shape[2], num_heads=8).to(device)
    input_tensor = torch.randn(input_shape, dtype=torch.float32, device=device)
    state_dict = OrderedDict()
    state_dict["net_state"] = atten.state_dict()
    state_dict["input_tensor"] = input_tensor
    output_tensor = atten(input_tensor)
    state_dict["output_tensor"] = output_tensor
    torch.save(state_dict, weight_path)
