import copy
import math
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from models.initializer import initialize_from_cfg
from torch import Tensor, nn

from models.reconstructions.dat import DAT

from fsp_net import Converter, GroupFusion, OutPut


class UniAD0(nn.Module):
    def __init__(
        self,
        inplanes,
        instrides,
        feature_size,
        feature_jitter,
        neighbor_mask,
        hidden_dim,
        pos_embed_type,
        save_recon,
        initializer,
        **kwargs,
    ):
        super().__init__()
        assert isinstance(inplanes, list) and len(inplanes) == 1
        assert isinstance(instrides, list) and len(instrides) == 1
        self.feature_size = feature_size
        self.num_queries = feature_size[0] * feature_size[1]
        self.feature_jitter = feature_jitter
        self.pos_embed = build_position_embedding(
            pos_embed_type, feature_size, hidden_dim
        )
        self.save_recon = save_recon

        self.transformer = Transformer(
            hidden_dim, feature_size, neighbor_mask, **kwargs
        )
        self.input_proj = nn.Linear(inplanes[0], hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, inplanes[0])

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=instrides[0])

        initialize_from_cfg(self, initializer)

    def add_jitter(self, feature_tokens, scale, prob):
        if random.uniform(0, 1) <= prob:
            num_tokens, batch_size, dim_channel = feature_tokens.shape
            feature_norms = (
                feature_tokens.norm(dim=2).unsqueeze(2) / dim_channel
            )  # (H x W) x B x 1
            jitter = torch.randn((num_tokens, batch_size, dim_channel)).cuda()
            jitter = jitter * feature_norms * scale
            feature_tokens = feature_tokens + jitter
        return feature_tokens

    def forward(self, input):
        feature_align = input["feature_align"]  # B x C X H x W
        feature_tokens = rearrange(
            feature_align, "b c h w -> (h w) b c"
        )  # (H x W) x B x C
        if self.training and self.feature_jitter:
            feature_tokens = self.add_jitter(
                feature_tokens, self.feature_jitter.scale, self.feature_jitter.prob
            )
        feature_tokens = self.input_proj(feature_tokens)  # (H x W) x B x C
        pos_embed = self.pos_embed(feature_tokens)  # (H x W) x C
        output_decoder, _ = self.transformer(
            feature_tokens, pos_embed
        )  # (H x W) x B x C
        feature_rec_tokens = self.output_proj(output_decoder)  # (H x W) x B x C
        feature_rec = rearrange(
            feature_rec_tokens, "(h w) b c -> b c h w", h=self.feature_size[0]
        )  # B x C X H x W

        if not self.training and self.save_recon:
            clsnames = input["clsname"]
            filenames = input["filename"]
            for clsname, filename, feat_rec in zip(clsnames, filenames, feature_rec):
                filedir, filename = os.path.split(filename)
                _, defename = os.path.split(filedir)
                filename_, _ = os.path.splitext(filename)
                save_dir = os.path.join(self.save_recon.save_dir, clsname, defename)
                os.makedirs(save_dir, exist_ok=True)
                feature_rec_np = feat_rec.detach().cpu().numpy()
                np.save(os.path.join(save_dir, filename_ + ".npy"), feature_rec_np)

        pred = torch.sqrt(
            torch.sum((feature_rec - feature_align) ** 2, dim=1, keepdim=True)
        )  # B x 1 x H x W
        pred = self.upsample(pred)  # B x 1 x H x W
        return {
            "feature_rec": feature_rec,
            "feature_align": feature_align,
            "pred": pred,
        }

class UniAD(nn.Module):
    def __init__(
        self,
        inplanes,
        instrides,
        feature_size,
        feature_jitter,
        neighbor_mask,
        hidden_dim,
        pos_embed_type,
        save_recon,
        initializer,
        **kwargs,
    ):
        super().__init__()
        assert isinstance(inplanes, list) and len(inplanes) == 1
        assert isinstance(instrides, list) and len(instrides) == 1
        self.feature_size = feature_size
        self.num_queries = feature_size[0] * feature_size[1]
        self.feature_jitter = feature_jitter
        self.pos_embed = build_position_embedding(
            pos_embed_type, feature_size, hidden_dim
        )
        self.save_recon = save_recon



        self.transformer = Transformer(
            hidden_dim, feature_size, neighbor_mask, **kwargs
        )
        self.input_proj = nn.Linear(inplanes[0], hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, inplanes[0])

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=instrides[0])

        initialize_from_cfg(self, initializer)

    def add_jitter(self, feature_tokens, scale, prob):
        if random.uniform(0, 1) <= prob:
            num_tokens, batch_size, dim_channel = feature_tokens.shape
            feature_norms = (
                feature_tokens.norm(dim=2).unsqueeze(2) / dim_channel
            )  # (H x W) x B x 1
            jitter = torch.randn((num_tokens, batch_size, dim_channel)).cuda()
            jitter = jitter * feature_norms * scale
            feature_tokens = feature_tokens + jitter
        return feature_tokens

    def forward(self, input):
        feature_align = input["feature_align"]  # B x C X H x W
        feature_tokens = rearrange(
            feature_align, "b c h w -> (h w) b c"
        )  # (H x W) x B x C
        if self.training and self.feature_jitter:
            feature_tokens = self.add_jitter(
                feature_tokens, self.feature_jitter.scale, self.feature_jitter.prob
            )
        feature_tokens = self.input_proj(feature_tokens)  # (H x W) x B x C
        pos_embed = self.pos_embed(feature_tokens)  # (H x W) x C


        output_decoder, _ = self.transformer(
            feature_tokens, pos_embed
        )  # (H x W) x B x C


        feature_rec_tokens = self.output_proj(output_decoder)  # (H x W) x B x C
        feature_rec = rearrange(
            feature_rec_tokens, "(h w) b c -> b c h w", h=self.feature_size[0]
        )  # B x C X H x W

        if not self.training and self.save_recon:
            clsnames = input["clsname"]
            filenames = input["filename"]
            for clsname, filename, feat_rec in zip(clsnames, filenames, feature_rec):
                filedir, filename = os.path.split(filename)
                _, defename = os.path.split(filedir)
                filename_, _ = os.path.splitext(filename)
                save_dir = os.path.join(self.save_recon.save_dir, clsname, defename)
                os.makedirs(save_dir, exist_ok=True)
                feature_rec_np = feat_rec.detach().cpu().numpy()
                np.save(os.path.join(save_dir, filename_ + ".npy"), feature_rec_np)

        pred = torch.sqrt(
            torch.sum((feature_rec - feature_align) ** 2, dim=1, keepdim=True)
        )  # B x 1 x H x W
        pred = self.upsample(pred)  # B x 1 x H x W
        return {
            "feature_rec": feature_rec,
            "feature_align": feature_align,
            "pred": pred,
        }

class Transformer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        feature_size,
        neighbor_mask,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.neighbor_mask = neighbor_mask
        dat_params =  {
    "upscale": 1,
    "in_chans": 256,
    "img_size": 224,
    "img_range": 1.0,
    "depth": [1, 1],  # Alternative: [6, 6, 6, 6, 6, 6]
    "embed_dim": 180,
    "num_heads": [3, 3],
    "expansion_factor": 2,
    "resi_connection": "1conv",
    "split_size": [8, 16]
    }
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(hidden_dim) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers,dat_params, encoder_norm
        )

        decoder_layer = TransformerDecoderLayer(
            hidden_dim,
            feature_size,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
        )
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.vit_chs = 256
        ###########
        self.group_converter_0 = Converter(dim_in=self.vit_chs, img_size=self.img_size)
        self.group_converter_1 = Converter(dim_in=self.vit_chs, img_size=self.img_size)
        self.group_converter_2 = Converter(dim_in=self.vit_chs, img_size=self.img_size)
        self.group_converter_3 = Converter(dim_in=self.vit_chs, img_size=self.img_size)
        self.group_converter_4 = Converter(dim_in=self.vit_chs, img_size=self.img_size)
        self.group_converter_5 = Converter(dim_in=self.vit_chs, img_size=self.img_size)

        #
        # self.gf1_1 = GroupFusion(256, 128)
        # self.gf1_2 = GroupFusion(256, 128)
        # self.gf1_3 = GroupFusion(256, 128)
        # self.gf1_4 = GroupFusion(256, 128)
        # self.gf1_5 = GroupFusion(256, 128)
        # self.gf1_6 = GroupFusion(256, 128, start=True)
        #
        # # Update next level to 128 -> 64 channels
        # self.gf2_1 = GroupFusion(128, 64)
        # self.gf2_2 = GroupFusion(128, 64)
        # self.gf2_3 = GroupFusion(128, 64, start=True)
        #
        # # Update final level to 64 -> 32 channels
        # self.gf3_1 = GroupFusion(64, 32)
        # self.gf3_2 = GroupFusion(64, 32, start=True)
        #
        # # Final fusion level for 32 channels
        # self.gf4_1 = GroupFusion(32, 32, start=True)
        #
        # # Update OutPut layers for new channel sizes and spatial scales
        # self.out1 = OutPut(in_chs=256, scale=16)
        # self.out2 = OutPut(in_chs=128, scale=8)
        # self.out3 = OutPut(in_chs=64, scale=4)
        # self.out4 = OutPut(in_chs=32)


    def group_converter_fn(self, tokens):
        group_converter_ls = [self.group_converter_0, self.group_converter_1, self.group_converter_2,
                              self.group_converter_3, self.group_converter_4, self.group_converter_5]
        tokens_ls = []
        for index in range(len(tokens) // 2):
            token_pair = [tokens[index * 2], tokens[index * 2 + 1]]
            token_pair_out = group_converter_ls[index](token_pair)
            tokens_ls.extend(token_pair_out)

        return tokens_ls


    def group_pyramid_decode(self, feature):
        # list 12x[8,768,24,24]
        # layer1 out
        f1_6_l, f2_6 = self.gf1_6(feature[-1], feature[-2])
        f1_5_l, f2_5 = self.gf1_5(torch.cat((feature[-3], f1_6_l), dim=1), feature[-4])
        f1_4_l, f2_4 = self.gf1_4(torch.cat((feature[-5], f1_5_l), dim=1), feature[-6])
        f1_3_l, f2_3 = self.gf1_3(torch.cat((feature[-7], f1_4_l), dim=1), feature[-8])
        f1_2_l, f2_2 = self.gf1_2(torch.cat((feature[-9], f1_3_l), dim=1), feature[-10])
        f1_1_o, f2_1 = self.gf1_1(torch.cat((feature[-11], f1_2_l), dim=1), feature[-12])  # f1_1_l [bs,768,24,24]
        # layer2 out
        f2_3_l, f3_3 = self.gf2_3(f2_6, f2_5)
        f2_2_l, f3_2 = self.gf2_2(torch.cat((f2_4, f2_3_l), dim=1), f2_3)
        f2_1_o, f3_1 = self.gf2_1(torch.cat((f2_2, f2_2_l), dim=1), f2_1)  # f2_1_l [bs,384,48,48]
        # layer3 out
        f3_2_l, f4_2 = self.gf3_2(f3_3, f3_2)
        f3_1_o, f4_1 = self.gf3_1(torch.cat((f3_2, f3_2_l), dim=1), f3_1)  # f3_1_l [bs,192,96,96]
        # layer4 out
        _, f5_1 = self.gf4_1(f4_2, f4_1)
        return f1_1_o, f2_1_o, f3_1_o, f5_1


    def pred_out(self, gpd_outs):
        return self.out1(gpd_outs[0]), self.out2(gpd_outs[1]), self.out3(gpd_outs[2]), self.out4(gpd_outs[3])
#################
    def generate_mask(self, feature_size, neighbor_size):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        h, w = feature_size
        hm, wm = neighbor_size
        mask = torch.ones(h, w, h, w)
        for idx_h1 in range(h):
            for idx_w1 in range(w):
                idx_h2_start = max(idx_h1 - hm // 2, 0)
                idx_h2_end = min(idx_h1 + hm // 2 + 1, h)
                idx_w2_start = max(idx_w1 - wm // 2, 0)
                idx_w2_end = min(idx_w1 + wm // 2 + 1, w)
                mask[
                    idx_h1, idx_w1, idx_h2_start:idx_h2_end, idx_w2_start:idx_w2_end
                ] = 0
        mask = mask.view(h * w, h * w)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
            .cuda()
        )
        return mask

    def forward(self, src, pos_embed):
        _, batch_size, _ = src.shape
        pos_embed = torch.cat(
            [pos_embed.unsqueeze(1)] * batch_size, dim=1
        )  # (H X W) x B x C

        if self.neighbor_mask:
            mask = self.generate_mask(
                self.feature_size, self.neighbor_mask.neighbor_size
            )
            mask_enc = mask if self.neighbor_mask.mask[0] else None
            mask_dec1 = mask if self.neighbor_mask.mask[1] else None
            mask_dec2 = mask if self.neighbor_mask.mask[2] else None
        else:
            mask_enc = mask_dec1 = mask_dec2 = None

        output_encoder, output_dat = self.encoder(
            src, mask=mask_enc, pos=pos_embed
        )  # (H X W) x B x C

        ##### go through NL-TEM ######

        # list 12x[8,576,768]
        y = output_encoder + output_dat
        feature = self.group_converter_fn(y)

        memory_list = feature




        output_decoder = self.decoder(
            output_encoder,
            tgt_mask=mask_dec1,
            memory_mask=mask_dec2,
            pos=pos_embed,
        )  # (H X W) x B x C

        return output_decoder, output_encoder

class TransformerEncoderLayerx(nn.Module):
    def __init__(
        self,
        hidden_dim,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()

        self.feature_size = [14,14]

        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        upscale = 1
        self.dat = DAT(
            upscale=upscale,
            in_chans=256,
            img_size=224,
            img_range=1.,
            depth=[1, 1],  # [6,6,6,6,6,6],
            embed_dim=180,
            num_heads=[3, 3],
            expansion_factor=2,
            resi_connection='1conv',
            split_size=[8, 16],
        )



    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        # print(src.shape,'src Shape********************post')
        input_dat = rearrange(
            src, "(h w) b c -> b c h w", h=self.feature_size[0]
        )  # B x C X H x W
        dat_out = self.dat(input_dat)
        dat_out = rearrange(
            dat_out, "b c h w -> (h w) b c"
        )  # (H x W) x B x C

        # src = src + self.dropout1(src2)
        src = dat_out + self.dropout1(src2)
        src = self.norm1(src)
        print(src.shape,'<<<<<<<<<<<<')
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

        # src2 = dat_out
        # src = src + self.dropout1(src2)
        # # dat_out = dat_out + self.dropout1(src)
        # src = self.norm1(src)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # src = src + self.dropout2(src2)
        # src = self.norm2(src)
        # return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        print(src.shape,'src Shape********************')
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            hidden_dim,
            nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
    ):
        super().__init__()

        self.feature_size = [14, 14]

        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        dim = hidden_dim
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1)
        )

        # self.aap = nn.AdaptiveAvgPool1d(180)



    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
            self,
            src,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        attened_x = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]

        # Apply pooling along the third dimension (256 -> 180)
        # src = self.aap(src.permute(0, 1, 2))  # Shape [196, 2, 180]
        #
        attened_x = rearrange(attened_x, "l b c -> b l c")

        L, B, C = src.shape
        H = 14
        W = 14
        src = rearrange(src, "(h w) b c -> b c h w", h=H, w=H)



        # convolution output
        conv_x = self.dwconv(src)

        # Adaptive Interaction Module (AIM)
        # C-Map (before sigmoid)
        channel_map = self.channel_interaction(conv_x).permute(0, 2, 3, 1).contiguous().view(B, 1, C) #2,1,256
        # S-Map (before sigmoid)
        attention_reshape = attened_x.transpose(-2, -1).contiguous().view(B, C, H, W)
        spatial_map = self.spatial_interaction(attention_reshape)



        # C-I
        attened_x = attened_x * torch.sigmoid(channel_map)
        # S-I
        conv_x = torch.sigmoid(spatial_map) * conv_x
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, L, C)

        x = attened_x + conv_x

        x = rearrange(x, "b l c -> l b c")
        src = rearrange(src, "b c h w -> (h w) b c", h=H, w=H)

        src = src + self.dropout1(x)
        # src = dat_out + self.dropout1(src2)
        src = self.norm1(src)
        # print(src.shape, '<<<<<<<<<<<<')
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)


        # print(f'Adaptive_Spatial_Attention{x.size()}')

        return src

    def forward_pre(
            self,
            src,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        print(src.shape, 'src Shape********************')
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
            self,
            src,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)



class TransformerEncoder0(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerEncoder1(nn.Module):
    def __init__(self, encoder_layer, num_layers, dat_params, norm=None, return_intermediate=True):
        super().__init__()
        # Transformer layers and DAT layers
        self.transformer_layers = _get_clones(encoder_layer, num_layers)
        self.dat_layers = nn.ModuleList([
            DAT(
                upscale=dat_params['upscale'],
                in_chans=dat_params['in_chans'],
                img_size=dat_params['img_size'],
                img_range=dat_params['img_range'],
                depth=[1],  # Setting depth to 1 for each DAT layer
                embed_dim=dat_params['embed_dim'],
                num_heads=dat_params['num_heads'],
                expansion_factor=dat_params['expansion_factor'],
                resi_connection=dat_params['resi_connection'],
                split_size=dat_params['split_size'],
            ) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        src,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ):
        transformer_intermediates = []
        dat_intermediates = []

        output = src

        # Loop through each Transformer and DAT layer pair
        for i in range(self.num_layers):
            # Process with Transformer layer
            output = self.transformer_layers[i](
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )
            transformer_intermediates.append(output)  # Collect intermediate Transformer output

        # Loop through each DAT layer pair
        for i in range(self.num_layers):
            # Process with Transformer layer
            if i == 0:
                dat_input = src + transformer_intermediates[i]
            else:
                dat_input = transformer_intermediates + dat_intermediates[i-1]

            dat_input = rearrange(dat_input, "(h w) b c -> b c h w", h=14, w=14)  # Adjust dimensions as needed
            dat_output = self.dat_layers[i](dat_input)  # Pass through corresponding DAT layer
            dat_output = rearrange(dat_output, "b c h w -> (h w) b c")  # Back to Transformer format
            dat_intermediates.append(dat_output)  # Collect intermediate DAT output


        # Apply normalization if specified on final Transformer output
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                transformer_intermediates[-1] = output  # Update last Transformer layer output

        # Return intermediate outputs from both the Transformer and DAT layers
        if self.return_intermediate:
            return transformer_intermediates, dat_intermediates

        return output

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, dat_params, norm=None, return_intermediate=True):
        super().__init__()
        # Transformer layers and DAT layers
        self.transformer_layers = _get_clones(encoder_layer, num_layers)
        self.dat_layers = nn.ModuleList([
            DAT(
                upscale=dat_params['upscale'],
                in_chans=dat_params['in_chans'],
                img_size=dat_params['img_size'],
                img_range=dat_params['img_range'],
                depth=[1],  # Setting depth to 1 for each DAT layer
                embed_dim=dat_params['embed_dim'],
                num_heads=dat_params['num_heads'],
                expansion_factor=dat_params['expansion_factor'],
                resi_connection=dat_params['resi_connection'],
                split_size=dat_params['split_size'],
            ) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        src,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
    ):
        transformer_intermediates = []
        dat_intermediates = []

        output = src

        # Loop through each paired Transformer and DAT layer
        for i in range(self.num_layers):
            # Process with Transformer layer
            output = self.transformer_layers[i](
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )
            transformer_intermediates.append(output)  # Collect intermediate Transformer output

            # Prepare input for DAT layer by combining Transformer output and previous DAT output
            if i == 0:
                dat_input = src + output  # Initial input for first DAT layer
            else:
                dat_input = transformer_intermediates[i] + dat_intermediates[i-1]  # Combine current Transformer and previous DAT output

            # Reshape for DAT layer
            dat_input = rearrange(dat_input, "(h w) b c -> b c h w", h=14, w=14)  # Adjust dimensions as needed
            dat_output = self.dat_layers[i](dat_input)  # Pass through corresponding DAT layer
            dat_output = rearrange(dat_output, "b c h w -> (h w) b c")  # Back to Transformer format
            dat_intermediates.append(dat_output)  # Collect intermediate DAT output

            # Prepare output for the next Transformer layer
            output = dat_output

        # Apply normalization if specified on final Transformer output
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                transformer_intermediates[-1] = output  # Update last Transformer layer output

        # Return intermediate outputs from both the Transformer and DAT layers
        if self.return_intermediate:
            return transformer_intermediates, dat_intermediates

        return output

class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        feature_size,
        nhead,
        dim_feedforward,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        num_queries = feature_size[0] * feature_size[1]
        self.learned_embed = nn.Embedding(num_queries, hidden_dim)

        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)

        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward(
        self,
        out,
        memory_list,
        layer_idx,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        # Select an intermediate memory feature from the encoder
        memory = memory_list[layer_idx % len(memory_list)]  # Cycle through layers if needed

        _, batch_size, _ = memory.shape
        tgt = self.learned_embed.weight
        tgt = torch.cat([tgt.unsqueeze(1)] * batch_size, dim=1)  # (H X W) x B x C

        # Self-attention
        tgt2 = self.self_attn(
            query=self.with_pos_embed(tgt, pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention with output
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, pos),
            key=self.with_pos_embed(out, pos),
            value=out,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feedforward network
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        memory_list,  # List of intermediate encoder features
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = memory_list[-1]  # Start with the last encoder output for initialization
        intermediate = []

        for idx, layer in enumerate(self.layers):
            output = layer(
                output,
                memory_list,  # Pass all intermediate features
                idx,  # Track the current layer index
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output

class TransformerDecoder0(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = memory

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output

class TransformerDecoderLayer0(nn.Module):
    def __init__(
        self,
        hidden_dim,
        feature_size,
        nhead,
        dim_feedforward,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.feature_size = feature_size
        num_queries = self.feature_size[0] * self.feature_size[1]
        self.learned_embed = nn.Embedding(num_queries, hidden_dim)  # (H x W) x C

        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        upscale = 1
        self.dat = DAT(
            upscale=upscale,
            in_chans=256,
            img_size=224,
            img_range=1.,
            depth=[2, 2],  # [6,6,6,6,6,6],
            embed_dim=180,
            num_heads=[3, 3],
            expansion_factor=2,
            resi_connection='1conv',
            split_size=[8, 16],
        )

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        out,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        _, batch_size, _ = memory.shape
        tgt = self.learned_embed.weight
        tgt = torch.cat([tgt.unsqueeze(1)] * batch_size, dim=1)  # (H X W) x B x C
############
        # tgt2 = self.self_attn(
        #     query=self.with_pos_embed(tgt, pos),
        #     key=self.with_pos_embed(memory, pos),
        #     value=memory,
        #     attn_mask=tgt_mask,
        #     key_padding_mask=tgt_key_padding_mask,
        # )[0]

        # print(tgt.shape,'src Shape********************post')
        input_dat = rearrange(
            tgt, "(h w) b c -> b c h w", h=self.feature_size[0]
        )  # B x C X H x W
        dat_out = self.dat(input_dat)
        dat_out = rearrange(
            dat_out, "b c h w -> (h w) b c"
        )  # (H x W) x B x C


        tgt2 = dat_out
        tgt = tgt + self.dropout1(tgt2)
        # tgt = dat_out + self.dropout1(tgt2)
###############
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, pos),
            key=self.with_pos_embed(out, pos),
            value=out,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        out,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        _, batch_size, _ = memory.shape
        tgt = self.learned_embed.weight
        tgt = torch.cat([tgt.unsqueeze(1)] * batch_size, dim=1)  # (H X W) x B x C

        tgt2 = self.norm1(tgt)
        tgt2 = self.self_attn(
            query=self.with_pos_embed(tgt2, pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        print(tgt.shape,'src Shape********************pre')

        tgt = tgt + self.dropout1(tgt2)

        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, pos),
            key=self.with_pos_embed(out, pos),
            value=out,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        out,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                out,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
            )
        return self.forward_post(
            out,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
        )


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self,
        feature_size,
        num_pos_feats=128,
        temperature=10000,
        normalize=False,
        scale=None,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        not_mask = torch.ones((self.feature_size[0], self.feature_size[1]))  # H x W
        y_embed = not_mask.cumsum(0, dtype=torch.float32)
        x_embed = not_mask.cumsum(1, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[-1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos_y = torch.stack(
            (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).flatten(0, 1)  # (H X W) X C
        return pos.to(tensor.device)


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, feature_size, num_pos_feats=128):
        super().__init__()
        self.feature_size = feature_size  # H, W
        self.row_embed = nn.Embedding(feature_size[0], num_pos_feats)
        self.col_embed = nn.Embedding(feature_size[1], num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor):
        i = torch.arange(self.feature_size[1], device=tensor.device)  # W
        j = torch.arange(self.feature_size[0], device=tensor.device)  # H
        x_emb = self.col_embed(i)  # W x C // 2
        y_emb = self.row_embed(j)  # H x C // 2
        pos = torch.cat(
            [
                torch.cat(
                    [x_emb.unsqueeze(0)] * self.feature_size[0], dim=0
                ),  # H x W x C // 2
                torch.cat(
                    [y_emb.unsqueeze(1)] * self.feature_size[1], dim=1
                ),  # H x W x C // 2
            ],
            dim=-1,
        ).flatten(
            0, 1
        )  # (H X W) X C
        return pos


def build_position_embedding(pos_embed_type, feature_size, hidden_dim):
    if pos_embed_type in ("v2", "sine"):
        # TODO find a better way of exposing other arguments
        pos_embed = PositionEmbeddingSine(feature_size, hidden_dim // 2, normalize=True)
    elif pos_embed_type in ("v3", "learned"):
        pos_embed = PositionEmbeddingLearned(feature_size, hidden_dim // 2)
    else:
        raise ValueError(f"not supported {pos_embed_type}")
    return pos_embed


class Model(nn.Module):
    def __init__(self, ckpt, img_size=224):
        super(Model, self).__init__()
        # self.encoder = vit.deit_base_distilled_patch16_384()
        # if ckpt is not None:
        #     ckpt = torch.load(ckpt, map_location='cpu')
        #     msg = self.encoder.load_state_dict(ckpt["model"], strict=False)
        #     print("====================================")
        #     print(msg)

        self.img_size = img_size
        self.vit_chs = 256

        img_size = 224
        vit_chs = 256
        group_converter_0 = Converter(dim_in=vit_chs, img_size=img_size)

        dat_params = {
            "upscale": 1,
            "in_chans": 256,
            "img_size": 224,
            "img_range": 1.0,
            "depth": [1, 1],  # Alternative: [6, 6, 6, 6, 6, 6]
            "embed_dim": 180,
            "num_heads": [3, 3],
            "expansion_factor": 2,
            "resi_connection": "1conv",
            "split_size": [8, 16]
        }
        hidden_dim = 256
        nhead = 8
        dim_feedforward = 2048
        dropout = 0.1
        activation = "relu"
        normalize_before = False
        num_encoder_layers = 4
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, dat_params, encoder_norm
        )

        self.group_converter_0 = Converter(dim_in=self.vit_chs, img_size=self.img_size)
        self.group_converter_1 = Converter(dim_in=self.vit_chs, img_size=self.img_size)
        self.group_converter_2 = Converter(dim_in=self.vit_chs, img_size=self.img_size)
        self.group_converter_3 = Converter(dim_in=self.vit_chs, img_size=self.img_size)
        self.group_converter_4 = Converter(dim_in=self.vit_chs, img_size=self.img_size)
        self.group_converter_5 = Converter(dim_in=self.vit_chs, img_size=self.img_size)

        # self.gf1_1 = GroupFusion(768, 384)
        # self.gf1_2 = GroupFusion(768, 384)
        # self.gf1_3 = GroupFusion(768, 384)
        # self.gf1_4 = GroupFusion(768, 384)
        # self.gf1_5 = GroupFusion(768, 384)
        # self.gf1_6 = GroupFusion(768, 384, start=True)
        #
        # self.gf2_1 = GroupFusion(384, 192)
        # self.gf2_2 = GroupFusion(384, 192)
        # self.gf2_3 = GroupFusion(384, 192, start=True)
        #
        # self.gf3_1 = GroupFusion(192, 192)
        # self.gf3_2 = GroupFusion(192, 192, start=True)
        #
        # self.gf4_1 = GroupFusion(192, 192, start=True)
        #
        # self.out1 = OutPut(in_chs=768, scale=16)
        # self.out2 = OutPut(in_chs=384, scale=8)
        # self.out3 = OutPut(in_chs=192, scale=4)
        # self.out4 = OutPut(in_chs=192)

        # Update GroupFusion to match new channel sizes
        self.gf1_1 = GroupFusion(256, 128)
        self.gf1_2 = GroupFusion(256, 128)
        self.gf1_3 = GroupFusion(256, 128)
        self.gf1_4 = GroupFusion(256, 128)
        self.gf1_5 = GroupFusion(256, 128)
        self.gf1_6 = GroupFusion(256, 128, start=True)

        # Update next level to 128 -> 64 channels
        self.gf2_1 = GroupFusion(128, 64)
        self.gf2_2 = GroupFusion(128, 64)
        self.gf2_3 = GroupFusion(128, 64, start=True)

        # Update final level to 64 -> 32 channels
        self.gf3_1 = GroupFusion(64, 32)
        self.gf3_2 = GroupFusion(64, 32, start=True)

        # Final fusion level for 32 channels
        self.gf4_1 = GroupFusion(32, 32, start=True)

        # Update OutPut layers for new channel sizes and spatial scales
        self.out1 = OutPut(in_chs=256, scale=16)
        self.out2 = OutPut(in_chs=128, scale=8)
        self.out3 = OutPut(in_chs=64, scale=4)
        self.out4 = OutPut(in_chs=32)

    def group_converter_fn(self, tokens):
        group_converter_ls = [self.group_converter_0, self.group_converter_1, self.group_converter_2,
                              self.group_converter_3, self.group_converter_4, self.group_converter_5]
        tokens_ls = []
        for index in range(len(tokens) // 2):
            token_pair = [tokens[index * 2], tokens[index * 2 + 1]]
            token_pair_out = group_converter_ls[index](token_pair)
            tokens_ls.extend(token_pair_out)

        return tokens_ls

    def group_pyramid_decode(self, feature):
        # list 12x[8,768,24,24]
        # layer1 out
        f1_6_l, f2_6 = self.gf1_6(feature[-1], feature[-2])
        f1_5_l, f2_5 = self.gf1_5(torch.cat((feature[-3], f1_6_l), dim=1), feature[-4])
        f1_4_l, f2_4 = self.gf1_4(torch.cat((feature[-5], f1_5_l), dim=1), feature[-6])
        f1_3_l, f2_3 = self.gf1_3(torch.cat((feature[-7], f1_4_l), dim=1), feature[-8])
        f1_2_l, f2_2 = self.gf1_2(torch.cat((feature[-9], f1_3_l), dim=1), feature[-10])
        f1_1_o, f2_1 = self.gf1_1(torch.cat((feature[-11], f1_2_l), dim=1), feature[-12])  # f1_1_l [bs,768,24,24]
        # layer2 out
        f2_3_l, f3_3 = self.gf2_3(f2_6, f2_5)
        f2_2_l, f3_2 = self.gf2_2(torch.cat((f2_4, f2_3_l), dim=1), f2_3)
        f2_1_o, f3_1 = self.gf2_1(torch.cat((f2_2, f2_2_l), dim=1), f2_1)  # f2_1_l [bs,384,48,48]
        # layer3 out
        f3_2_l, f4_2 = self.gf3_2(f3_3, f3_2)
        f3_1_o, f4_1 = self.gf3_1(torch.cat((f3_2, f3_2_l), dim=1), f3_1)  # f3_1_l [bs,192,96,96]
        # layer4 out
        _, f5_1 = self.gf4_1(f4_2, f4_1)
        return f1_1_o, f2_1_o, f3_1_o, f5_1

    def pred_out(self, gpd_outs):
        return self.out1(gpd_outs[0]), self.out2(gpd_outs[1]), self.out3(gpd_outs[2]), self.out4(gpd_outs[3])

    def forward(self, input):

        feature_align = input["feature_align"]  # B x C X H x W
        feature_tokens = rearrange(
            feature_align, "b c h w -> (h w) b c"
        )  # (H x W) x B x C
        if self.training and self.feature_jitter:
            feature_tokens = self.add_jitter(
                feature_tokens, self.feature_jitter.scale, self.feature_jitter.prob
            )
        feature_tokens = self.input_proj(feature_tokens)  # (H x W) x B x C
        pos_embed = self.pos_embed(feature_tokens)  # (H x W) x C
        output_decoder = self.encoder(feature_tokens,pos_embed) # B x C x H x W

        output_decoder, _ = self.transformer(
            feature_tokens, pos_embed
        )  # (H x W) x B x C
        feature_rec_tokens = self.output_proj(output_decoder)  # (H x W) x B x C
        feature_rec = rearrange(
            feature_rec_tokens, "(h w) b c -> b c h w", h=self.feature_size[0]
        )  # B x C X H x W

        if not self.training and self.save_recon:
            clsnames = input["clsname"]
            filenames = input["filename"]
            for clsname, filename, feat_rec in zip(clsnames, filenames, feature_rec):
                filedir, filename = os.path.split(filename)
                _, defename = os.path.split(filedir)
                filename_, _ = os.path.splitext(filename)
                save_dir = os.path.join(self.save_recon.save_dir, clsname, defename)
                os.makedirs(save_dir, exist_ok=True)
                feature_rec_np = feat_rec.detach().cpu().numpy()
                np.save(os.path.join(save_dir, filename_ + ".npy"), feature_rec_np)

        pred = torch.sqrt(
            torch.sum((feature_rec - feature_align) ** 2, dim=1, keepdim=True)
        )  # B x 1 x H x W
        pred = self.upsample(pred)  # B x 1 x H x W
        return {
            "feature_rec": feature_rec,
            "feature_align": feature_align,
            "pred": pred,
        }

        # B Seq
        B, C, H, W = img.size()
        x = self.encoder(img)  # list 12x[8,576,768]
        y = x[0] + [1]
        feature = self.group_converter_fn(y)
        gpd_outs = self.group_pyramid_decode(feature)
        return self.pred_out(gpd_outs)

if __name__ == '__main__':
    # # Test the TransformerEncoderLayer
    #     # Define input parameters
    #     hidden_dim = 256
    #     nhead = 8
    #     batch_size = 2
    #     seq_len = 196  # Assume 14x14 sequence length
    #     feature_dim = 256  # Should match hidden_dim
    #
    #     # Create an instance of TransformerEncoderLayer
    #     encoder_layer = TransformerEncoderLayer(hidden_dim, nhead)
    #
    #     # Generate a random input tensor with shape (seq_len, batch_size, hidden_dim)
    #     src = torch.rand(seq_len, batch_size, feature_dim)
    #
    #     # Run the forward pass
    #     output = encoder_layer(src)
    #
    #     print("Output shape:", output.shape)
    #     # print("Output:", output)


    # img_size = 384
    # vit_chs = 768


    img_size = 224
    vit_chs = 256
    group_converter_0 = Converter(dim_in=vit_chs, img_size=img_size)

    dat_params = {
        "upscale": 1,
        "in_chans": 256,
        "img_size": 224,
        "img_range": 1.0,
        "depth": [1, 1],  # Alternative: [6, 6, 6, 6, 6, 6]
        "embed_dim": 180,
        "num_heads": [3, 3],
        "expansion_factor": 2,
        "resi_connection": "1conv",
        "split_size": [8, 16]
    }
    hidden_dim = 256
    nhead = 8
    dim_feedforward = 2048
    dropout = 0.1
    activation = "relu"
    normalize_before = False
    num_encoder_layers = 4
    encoder_layer = TransformerEncoderLayer(
        hidden_dim, nhead, dim_feedforward, dropout, activation, normalize_before
    )
    encoder_norm =  None
    encoder = TransformerEncoder(
        encoder_layer, num_encoder_layers, dat_params, encoder_norm
    )

    x = torch.randn((196, 2, 256), dtype=torch.float32)
    y = encoder(x)
    z = y[0] + y[1]

    input = [rearrange(i, "c b d-> b c d") for i in z]

    print(input[0].shape)
    out = group_converter_0(input)
    print(out[0].shape)
    # out = Model.group_pyramid_decode(fature = out)
    # gpd_outs = Model.group_pyramid_decode(feature)
    # pred = Model.pred_out(gpd_outs)

    # # print(y)
