import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from .build import MODELS
import random
from extensions.chamfer_dist import ChamferDistanceL2

from utils.logger import *
from .modules import *


# Hierarchical Encoder
class H_Encoder(nn.Module):

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.encoder_depths = config.encoder_depths
        self.encoder_dims =  config.encoder_dims
        self.local_radius = config.local_radius # NOTE the local attention ...

        # token merging and positional embeddings
        self.token_embed = nn.ModuleList()
        self.encoder_pos_embeds = nn.ModuleList()
        for i in range(len(self.encoder_dims)):
            if i == 0:
                self.token_embed.append(Token_Embed(in_c=3, out_c=self.encoder_dims[i]))
            else:
                self.token_embed.append(Token_Embed(in_c=self.encoder_dims[i - 1], out_c=self.encoder_dims[i]))
            
            self.encoder_pos_embeds.append(nn.Sequential(
                            nn.Linear(3, self.encoder_dims[i]),
                            nn.GELU(),
                            nn.Linear(self.encoder_dims[i], self.encoder_dims[i]),
                        ))

        # encoder blocks
        self.encoder_blocks = nn.ModuleList()

        depth_count = 0
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(self.encoder_depths))]
        for i in range(len(self.encoder_depths)):
            self.encoder_blocks.append(Encoder_Block(
                            embed_dim=self.encoder_dims[i],
                            depth=self.encoder_depths[i],
                            drop_path_rate=dpr[depth_count: depth_count + self.encoder_depths[i]],
                            num_heads=config.num_heads,
                        ))
            depth_count += self.encoder_depths[i]

        # self.encoder_norms = nn.ModuleList()
        # for i in range(len(self.encoder_depths)):
        #     self.encoder_norms.append(nn.LayerNorm(self.encoder_dims[i]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def local_att_mask(self, xyz, radius, dist=None):
        with torch.no_grad():
            if dist is None or dist.shape[1] != xyz.shape[1]:
                dist = torch.cdist(xyz, xyz, p=2)
            # entries that are True in the mask do not contribute to self-attention
            # so points outside the radius are not considered
            mask = dist >= radius
        return mask, dist

    def forward(self, neighborhoods, centers, idxs, eval=False):
        # hierarchical encoding
        xyz_dist = None
        for i in range(len(centers)):
            # 1st-layer encoder, conduct token embedding
            if i == 0:
                group_input_tokens = self.token_embed[i](neighborhoods[0])
            # intermediate layers, conduct token merging
            else:
                b, g1, _ = x_vis.shape
                b, g2, k2, _ = neighborhoods[i].shape
                x_vis_neighborhoods = x_vis.reshape(b * g1, -1)[idxs[i], :].reshape(b, g2, k2, -1)
                group_input_tokens = self.token_embed[i](x_vis_neighborhoods)
            
            if self.local_radius[i] > 0:
                mask_radius, xyz_dist = self.local_att_mask(centers[i], self.local_radius[i], xyz_dist)
                mask_vis_att = mask_radius 
            else:
                mask_vis_att = None

            pos = self.encoder_pos_embeds[i](centers[i])

            x_vis = self.encoder_blocks[i](group_input_tokens, pos, mask_vis_att)
        return x_vis


@MODELS.register_module()
class Point_M2AE_ModelNet40(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_M2AE_ModelNet40]', logger ='Point_M2AE_ModelNet40')
        self.config = config
        
        # tokenizers
        self.group_sizes = config.group_sizes
        self.num_groups = config.num_groups
        self.feat_dim = config.encoder_dims[-1]
        self.group_dividers = nn.ModuleList()
        for i in range(len(self.group_sizes)):
            self.group_dividers.append(Group(num_group=self.num_groups[i], group_size=self.group_sizes[i]))

        # hierarchical encoder
        self.h_encoder = H_Encoder(config)

        # cls head
        self.norm = nn.LayerNorm(self.feat_dim)
        self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.feat_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 40)
        )

        # for smooth loss
        self.smooth = config.smooth

    def load_model_from_ckpt(self, ckpt_path):
        state_dict = torch.load(ckpt_path)
        incompatible = self.load_state_dict(state_dict['base_model'], strict=False)
        if incompatible.missing_keys:
            print_log('missing_keys', logger='Point_M2AE_ModelNet40')
            print_log(
                get_missing_parameters_message(incompatible.missing_keys),
                logger='Point_M2AE_ModelNet40'
            )
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger='Point_M2AE_ModelNet40')
            print_log(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                logger='Point_M2AE_ModelNet40'
            )

    def get_loss_acc(self, ret, gt):
        loss = self.smooth_loss(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def smooth_loss(self, pred, gt):
        eps = self.smooth
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gt.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
        return loss

    def forward(self, pts, **kwargs):
        # multi-scale representations of point clouds
        neighborhoods, centers, idxs = [], [], []
        for i in range(len(self.group_dividers)):
            if i == 0:
                neighborhood, center, idx = self.group_dividers[i](pts)
            else:
                neighborhood, center, idx = self.group_dividers[i](center)
            neighborhoods.append(neighborhood)
            centers.append(center)
            idxs.append(idx)  # neighbor indices

        # hierarchical encoder
        x_vis = self.h_encoder(neighborhoods, centers, idxs, eval=True)

        # classifier head
        x_vis = self.norm(x_vis)
        concat_f = x_vis.mean(1) + x_vis.max(1)[0]
        ret = self.cls_head_finetune(concat_f)
        return ret

@MODELS.register_module()
class Point_M2AE_ScanObjectNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_M2AE_ScanObjectNN]', logger ='Point_M2AE_ScanObjectNN')
        self.config = config
        
        # tokenizers
        self.group_sizes = config.group_sizes
        self.num_groups = config.num_groups
        self.feat_dim = config.encoder_dims[-1]
        self.group_dividers = nn.ModuleList()
        for i in range(len(self.group_sizes)):
            self.group_dividers.append(Group(num_group=self.num_groups[i], group_size=self.group_sizes[i]))

        # hierarchical encoder
        self.h_encoder = H_Encoder(config)

        # cls head
        self.norm = nn.LayerNorm(self.feat_dim)
        self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.feat_dim * 2, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 15)
        )

        # for cross entropy loss
        self.loss_ce = nn.CrossEntropyLoss()

    def load_model_from_ckpt(self, ckpt_path):
        state_dict = torch.load(ckpt_path)
        incompatible = self.load_state_dict(state_dict['base_model'], strict=False)
        if incompatible.missing_keys:
            print_log('missing_keys', logger='Point_M2AE_ScanObjectNN')
            print_log(
                get_missing_parameters_message(incompatible.missing_keys),
                logger='Point_M2AE_ScanObjectNN'
            )
        if incompatible.unexpected_keys:
            print_log('unexpected_keys', logger='Point_M2AE_ScanObjectNN')
            print_log(
                get_unexpected_parameters_message(incompatible.unexpected_keys),
                logger='Point_M2AE_ScanObjectNN'
            )

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def forward(self, pts, **kwargs):
        # multi-scale representations of point clouds
        neighborhoods, centers, idxs = [], [], []
        for i in range(len(self.group_dividers)):
            if i == 0:
                neighborhood, center, idx = self.group_dividers[i](pts)
            else:
                neighborhood, center, idx = self.group_dividers[i](center)
            neighborhoods.append(neighborhood)
            centers.append(center)
            idxs.append(idx)  # neighbor indices

        # hierarchical encoder
        x_vis = self.h_encoder(neighborhoods, centers, idxs, eval=True)
        
        # classifier head
        x_vis = self.norm(x_vis)
        concat_f = torch.cat((x_vis.mean(1), x_vis[:, 1:].max(1)[0]), dim=1)
        ret = self.cls_head_finetune(concat_f)
        return ret

