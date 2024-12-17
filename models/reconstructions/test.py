import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch import Tensor
from torch.nn import functional as F

from timm.models.layers import DropPath, trunc_normal_
from einops.layers.torch import Rearrange
from einops import rearrange

import math
import numpy as np

from models.reconstructions.dat import Adaptive_Channel_Attention, Adaptive_Spatial_Attention
from models.reconstructions.uniad import TransformerEncoderLayer


# def _get_clones(module, N):
#     # return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
#
#     layers = nn.ModuleList()
#     for i in range(N):
#         # encoder_layer = TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation, normalize_before, index = i)
#         layer = module(hidden_dim, nhead, dim_feedforward, dropout, activation, normalize_before, index = i)
#         l = module(i)
#         layers.append(layer)
#     return layers

# if __name__ == "__main__":
#     dim = 180
#     num_heads=2
#     reso=64
#     split_size=[2, 4]
#     shift_size=[1, 2]
#     expansion_factor=4.
#     qkv_bias=False
#     qk_scale=None
#     drop=0.
#     attn_drop=0.
#     drop_path=0.
#     act_layer=nn.GELU
#     norm_layer=nn.LayerNorm
#     rg_idx=0
#     b_idx=0
#
#     model = nn.ModuleList()
#
#
#      # self.norm1 = norm_layer(hidden_dim)s
#     for b_idx in range(1,5):
#         if b_idx % 2 == 0:
#             # DSTB
#             attn = Adaptive_Spatial_Attention(
#                 dim, num_heads=num_heads, reso=reso, split_size=split_size, shift_size=shift_size,
#                 qkv_bias=qkv_bias,
#                 qk_scale=qk_scale,
#                 drop=drop, attn_drop=attn_drop, rg_idx=rg_idx, b_idx=b_idx
#             )
#         else:
#             # DCTB
#             attn = Adaptive_Channel_Attention(
#                 dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
#                 proj_drop=drop
#             )
#         model.append(attn)
#
#     # print(model)
#     model.cuda()
#     x = torch.randn((2, 196, 180), dtype=torch.float32).cuda()
#     # Pass the input through each layer in the model
#     for layer in model:
#         x = layer(x,14,14)
#
#
#     print(x.shape,'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
if __name__ == "__main__":

    import json

# Function to load a JSON file with multiple JSON objects
    def load_multi_json(filename):
        data = []
        with open(filename, 'r') as file:
            for line in file:
                if line.strip():  # Skip empty lines
                    data.append(json.loads(line))
        return data

    # Load test.json
    test_data = load_multi_json('../../data/MVTec-AD/test.json')

    # Load train.json
    train_data = load_multi_json('../../data/MVTec-AD/train.json')
    label_0_entries = [entry for entry in test_data if entry['label'] == 0]

    # Filter the entries with label=0 in test.json

    selected_classes = ['capsule','screw','pill']
    # label_0_entries = [entry for entry in test_data if entry['label'] == 0]
    label_0_entries = [entry for entry in test_data if entry['label'] == 0 and (entry['clsname'] in selected_classes)]
    # Append these entries to the train data
    combined_data = train_data + label_0_entries

    # Save the combined data to a new JSON file
    with open('../../data/MVTec-AD/combined_train_test.json', 'w') as combined_file:
        for entry in combined_data:
            json.dump(entry, combined_file)
            combined_file.write('\n')

    print(f"Merged {len(label_0_entries)} entries from test.json to train.json.")
#
#     # counting number of images
#     class_dict = {
#     "bottle": 0,
#     "cable": 0,
#     "capsule": 0,
#     "carpet": 0,
#     "grid": 0,
#     "hazelnut": 0,
#     "leather": 0,
#     "metal_nut": 0,
#     "pill": 0,
#     "screw": 0,
#     "tile": 0,
#     "toothbrush": 0,
#     "transistor": 0,
#     "wood": 0,
#     "zipper": 0
# }
#     for entry in label_0_entries:
#         class_dict[entry['filename'].split('/')[0]] += 1
#     print(class_dict)


# {'bottle': 209, 'cable': 224, 'capsule': 219, 'carpet': 280, 'grid': 264, 'hazelnut': 391, 'leather': 245, 'metal_nut': 220, 'pill': 267, 'screw': 320, 'tile': 230, 'toothbrush': 60, 'transistor': 213, 'wood': 247, 'zipper': 240}
# {'bottle': 20, 'cable': 58, 'capsule': 23, 'carpet': 28, 'grid': 21, 'hazelnut': 40, 'leather': 32, 'metal_nut': 22, 'pill': 26, 'screw': 41, 'tile': 33, 'toothbrush': 12, 'transistor': 60, 'wood': 19, 'zipper': 32}
