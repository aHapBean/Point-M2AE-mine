import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_
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

        self.mask_ratio = config.mask_ratio 
        self.encoder_depths = config.encoder_depths
        self.encoder_dims =  config.encoder_dims
        self.local_radius = config.local_radius

        # token merging and positional embeddings
        self.token_embed = nn.ModuleList()
        self.encoder_pos_embeds = nn.ModuleList()
        self.increase_dims = nn.ModuleList()
        for i in range(len(self.encoder_dims)):
            if i == 0:
                self.token_embed.append(Token_Embed(in_c=3, out_c=self.encoder_dims[i]))
                self.increase_dims.append(nn.Linear(self.encoder_dims[i], self.encoder_dims[i], bias=True)) # Useless only as 占位符
            else:
                self.token_embed.append(Token_Embed(in_c=self.encoder_dims[i - 1], out_c=self.encoder_dims[i]))     # Token_Embed is lightweight pointnet
            
                # mine NOTE learn from code of MAE
                self.increase_dims.append(nn.Linear(self.encoder_dims[i - 1], self.encoder_dims[i], bias=True))
            
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

        self.encoder_norms = nn.ModuleList()
        for i in range(len(self.encoder_depths)):
            self.encoder_norms.append(nn.LayerNorm(self.encoder_dims[i]))

        self.apply(self._init_weights)
        
        self.mask_point_token = nn.Parameter(torch.zeros(1, 1, self.encoder_dims[0]))
        self.mask_pos_token = nn.Parameter(torch.zeros(1, 1, self.encoder_dims[0]))
        trunc_normal_(self.mask_point_token, std=.02)
        trunc_normal_(self.mask_pos_token, std=.02)

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

    def rand_mask(self, center):
        B, G, _ = center.shape
        self.num_mask = int(self.mask_ratio * G)
        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G-self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)
        return overall_mask.to(center.device) # B G

    def local_att_mask(self, xyz, radius, dist=None):
        with torch.no_grad():
            if dist is None or dist.shape[1] != xyz.shape[1]:   # (B, G, G)
                dist = torch.cdist(xyz, xyz, p=2)
            # entries that are True in the mask do not contribute to self-attention
            # so points outside the radius are not considered
            mask = dist >= radius
        return mask, dist

    def forward(self, neighborhoods, centers, idxs, eval=False):
        # generate mask at the highest level
        bool_masked_pos = []
        if eval:
            # no mask
            B, G, _ = centers[-1].shape
            bool_masked_pos.append(torch.zeros(B, G).bool().cuda())
        else:
            # mask_index: 1, mask; 0, vis
            bool_masked_pos.append(self.rand_mask(centers[-1]))
        
        # print(f'bool_masked_pos: {bool_masked_pos[-1].long().sum(dim=1)}')  # 128, 64 # bool_masked_pos: tensor([51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51,
        # print((~(bool_masked_pos[-1])).long().sum(dim=1)) # tensor([13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 
        
        # Multi-scale Masking by back-propagation
        # NOTE here ????
        # print(bool_masked_pos[-1].shape)    # 128, 64 -> B, G
        for i in range(len(neighborhoods) - 1, 0, -1):  # 倒序， i 取的是 i - 1 中center的点 
            b, g, k, _ = neighborhoods[i].shape
            idx = idxs[i].reshape(b * g, -1)    # idxs存的是选取点的索引 B, G, M
            # print(f'idx: {idx.shape}')  # 8192 (128 * 64), 8
            # print(idx)    # 值域是 0- 256 * 128
            # exit(-1)
            idx_masked = ~(bool_masked_pos[-1].reshape(-1).unsqueeze(-1)) * idx # B * G, -1 with B * G, -1 -> B * G, M
            # 这里留下了center 
            # idx_masked -> B * G, M
            
            # print((~(bool_masked_pos[-1])).long().sum(dim=1))  # tensor([13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
            # print(idx_masked)
            # exit(-1)
            
            
            # 这是没有被mask的点
            idx_masked = idx_masked.reshape(-1).long()      
            masked_pos = torch.ones(b * centers[i - 1].shape[1]).cuda().scatter(0, idx_masked, 0).bool()    # scatter 是插入值到指定位置
            # 由于 idx_masked 的不同batch的patch重叠度不同，所以带来了长度的改变，反正有对应patch的center就被mask掉
            # 下一个scale的masked patch的点在这个scale全部被mask掉
            
            # 注意下一个的patch&centers点全部取自现在的center
            
            # 是一个非常重要的操作，它将 idx_masked 中指定的位置的值设置为 0，其余位置保持为 1。 FIXME 这个怎么有点奇怪
            # idx没问题，后来的scale的idx都是这个idx_masked里面的东西
            # torch.scatter(dim, index, src)
            # print(f'bool_masked_pos: {masked_pos.reshape(b, centers[i - 1].shape[1]).long().sum(dim=1)}')   # bool_masked_pos: tensor([174, 166, 166, 172, 160, 162, 161, 163, 167, 174, 174, 162, 162, 161,  
            
            bool_masked_pos.append(masked_pos.reshape(b, centers[i - 1].shape[1]))  # 要得到 i 的 直接 centers[i - 1][masked_pos]即可
        
        # visible 的 idx
        # exit(-1)
        
        # hierarchical encoding
        bool_masked_pos.reverse()
        x_vis_list = []
        mask_vis_list = []
        xyz_dist = None
        B = neighborhoods[0].shape[0]
        # for patch matching
        pos_emd_mask = self.encoder_pos_embeds[0](centers[-1][bool_masked_pos[-1]]).reshape(B, -1, self.encoder_dims[0])
        _, N, _ = pos_emd_mask.shape
        mask_point_token = self.mask_point_token.expand(B, N, -1)  
        
        point_emd_mask = self.token_embed[0](neighborhoods[-1][bool_masked_pos[-1]].reshape(B, N, -1, 3))    # B, G, M, 3 -> B
        mask_pos_token = self.mask_pos_token.expand(B, N, -1)
        
        # Version 1:
        # no token propagation 直接分别 token embed
        
        # Version 2:
        # token propagation  
        
        for i in range(len(centers)):
            # 1st-layer encoder, conduct token embedding
            if i == 0:
                group_input_tokens = self.token_embed[i](neighborhoods[0])  # token embedding here
            # intermediate layers, conduct token merging
            else:
                b, g1, _ = x_vis.shape      # x_vis is the output from the previous layer
                b, g2, k2, _ = neighborhoods[i].shape
                x_vis_neighborhoods = x_vis.reshape(b * g1, -1)[idxs[i], :].reshape(b, g2, k2, -1)  # 取 all tokens from the previous layer 然后继续encode ??? !!!
                # idx 存的是 all tokens (当前scale)
                group_input_tokens = self.token_embed[i](x_vis_neighborhoods)
                # print(x_mask_without_pos)
                # print(self.increase_dims[i])
                # print(x_mask_without_pos.shape, ' pre')
                x_mask_without_pos = self.increase_dims[i](x_mask_without_pos)
                mask_pos_token = self.increase_dims[i](mask_pos_token)
                mask_point_token = self.increase_dims[i](mask_point_token)
                # print(x_mask_without_pos.shape, ' after')
                x_mask_without_point = self.increase_dims[i](x_mask_without_point)
                # NOTE
                # x_mask_neighborhoods = x_vis.reshape(b * g1, -1)[~idxs[i], :].reshape(b, g2, k2, -1)  # 取 masked tokens from the previous layer
            
            # visible_index
            bool_vis_pos = ~(bool_masked_pos[i])
            batch_size, seq_len, C = group_input_tokens.size()

            # Due to Multi-scale Masking different, samples of a batch have varying numbers of visible tokens
            # find the longest visible sequence in the batch
            vis_tokens_len = bool_vis_pos.long().sum(dim=1)     # 有不同长度的vis token
            max_tokens_len = torch.max(vis_tokens_len)
            # print(vis_tokens_len)   # tensor([264, 283, 294, 271, 269, 304, 319, 309, 281, 250, 266, 310, 304, 284, FIXME 为什么这里长度不同呢
            # raise ValueError
            # use the longest length (max_tokens_len) to construct tensors
            x_vis = torch.zeros(batch_size, max_tokens_len, C).cuda()
            masked_center = torch.zeros(batch_size, max_tokens_len, 3).cuda()       # visible centers
            mask_vis = torch.ones(batch_size, max_tokens_len, max_tokens_len).cuda()    # mask for attention ??? NOTE
            
            for bz in range(batch_size):
                # inject valid visible tokens
                vis_tokens = group_input_tokens[bz][bool_vis_pos[bz]]   # ?? NOTE bool_vis_pos[bz] 不是那个 idx吗
                x_vis[bz][0: vis_tokens_len[bz]] = vis_tokens           # 赋值 x_vis
                
                # inject valid visible centers
                vis_centers = centers[i][bz][bool_vis_pos[bz]]
                masked_center[bz][0: vis_tokens_len[bz]] = vis_centers
                
                # the mask for valid visible tokens/centers
                mask_vis[bz][0: vis_tokens_len[bz], 0: vis_tokens_len[bz]] = 0  # 
            
            if self.local_radius[i] > 0:
                # raise ValueError("????")  # 会过这个地方 不是说 disable for pre-training 吗 QQQ NOTE
                mask_radius, xyz_dist = self.local_att_mask(masked_center, self.local_radius[i], xyz_dist)  # B, G, C
                # disabled for pre-training, this step would not change mask_vis by *
                mask_vis_att = mask_radius * mask_vis
            else:
                mask_vis_att = mask_vis

            pos = self.encoder_pos_embeds[i](masked_center)
            
            # print(f'{point_emd_mask.shape} {mask_pos_token.shape} {mask_point_token.shape} {pos_emd_mask.shape}')   # torch.Size([128, 8, 96]) torch.Size([128, 51, 96]) torch.Size([128, 51, 96]) torch.Size([128, 51, 96])
            if i == 0:
                x_vis, x_mask_without_pos, x_mask_without_point = self.encoder_blocks[i](x_vis, pos, mask_vis_att, point_emd_mask, mask_pos_token, mask_point_token, pos_emd_mask, orig_mask=mask_vis)        # encoder block forward NOTE
            else:
                x_vis, x_mask_without_pos, x_mask_without_point = self.encoder_blocks[i](x_vis, pos, mask_vis_att, x_mask_without_pos, mask_pos_token, mask_point_token, x_mask_without_point, orig_mask=mask_vis)        # encoder block forward NOTE
            x_vis_list.append(x_vis)
            mask_vis_list.append(~(mask_vis[:, :, 0].bool()))       # mask_vis_list is the mask for attention

            if i == len(centers) - 1:
                pass
            else:
                group_input_tokens[bool_vis_pos] = x_vis[~(mask_vis[:, :, 0].bool())]       # 把可见的copy过来
                x_vis = group_input_tokens

        for i in range(len(x_vis_list)):
            x_vis_list[i] = self.encoder_norms[i](x_vis_list[i])

        return x_vis_list, x_mask_without_pos, x_mask_without_point


@MODELS.register_module()
class Point_M2AE(nn.Module):

    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_M2AE]', logger ='Point_M2AE')
        self.config = config
        
        # tokenizers
        self.group_sizes = config.group_sizes
        self.num_groups = config.num_groups
        self.group_dividers = nn.ModuleList()
        for i in range(len(self.group_sizes)):
            self.group_dividers.append(Group(num_group=self.num_groups[i], group_size=self.group_sizes[i]))

        # hierarchical encoder
        self.h_encoder = H_Encoder(config)

        # hierarchical decoder
        self.decoder_depths = config.decoder_depths
        self.decoder_dims = config.decoder_dims
        self.decoder_up_blocks = config.decoder_up_blocks

        self.encoder_dims = config.encoder_dims
        
        self.mask_token = nn.Parameter(torch.zeros(1, self.decoder_dims[0]))
        trunc_normal_(self.mask_token, std=.02)

        # self.h_decoder = nn.ModuleList()
        # self.decoder_pos_embeds = nn.ModuleList()
        # self.token_prop = nn.ModuleList()
        

        # depth_count = 0
        # dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(self.decoder_depths))]
        # for i in range(0, len(self.decoder_dims)):
        #     # decoder block
        #     self.h_decoder.append(Decoder_Block(
        #                 embed_dim=self.decoder_dims[i],
        #                 depth=self.decoder_depths[i],
        #                 drop_path_rate=dpr[depth_count: depth_count + self.decoder_depths[i]],
        #                 num_heads=config.num_heads,
        #             ))
        #     depth_count += self.decoder_depths[i]
        #     # decoder's positional embeddings
        #     self.decoder_pos_embeds.append(nn.Sequential(
        #                 nn.Linear(3, self.decoder_dims[i]),
        #                 nn.GELU(),
        #                 nn.Linear(self.decoder_dims[i], self.decoder_dims[i])
        #             ))
        #     # token propagation
        #     if i > 0:
        #         self.token_prop.append(PointNetFeaturePropagation(
        #                         self.decoder_dims[i] + self.decoder_dims[i - 1], self.decoder_dims[i],
        #                         blocks=self.decoder_up_blocks[i - 1], groups=1, res_expansion=1.0, bias=True
        #                     ))  
        # self.decoder_norm = nn.LayerNorm(self.decoder_dims[-1])
            
        # prediction head
        # self.rec_head = nn.Conv1d(self.decoder_dims[-1], 3 * self.group_sizes[0], 1)
        # loss
        self.rec_loss = ChamferDistanceL2().cuda()
        

        self.pred_pos_proj = nn.Sequential( # input B, M, C  
            nn.Linear(self.encoder_dims[-1], self.encoder_dims[-1]),
            nn.LayerNorm(self.encoder_dims[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoder_dims[-1], self.encoder_dims[-1]),
        )  
        
        self.pred_point_proj = nn.Sequential( # input B, M, C  
            nn.Linear(self.encoder_dims[-1], self.encoder_dims[-1]),
            nn.LayerNorm(self.encoder_dims[-1]),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoder_dims[-1], self.encoder_dims[-1]),
        )              

    def forward(self, pts, eval=False, **kwargs):
        # multi-scale representations of point clouds
        neighborhoods, centers, idxs = [], [], []
        # print(pts.shape)    # torch.Size([128, 2048, 3])
        # raise ValueError
        for i in range(len(self.group_dividers)):
            if i == 0:
                neighborhood, center, idx = self.group_dividers[i](pts)
            else:
                neighborhood, center, idx = self.group_dividers[i](center)
            # print(f'i: {i}, neighborhood: {neighborhood.shape}, center: {center.shape}, idx: {idx.shape}')
            # i: 0, neighborhood: torch.Size([128, 512, 16, 3]), center: torch.Size([128, 512, 3]), idx: torch.Size([1048576])    B, G, k, 3                                                  
            # i: 1, neighborhood: torch.Size([128, 256, 8, 3]), center: torch.Size([128, 256, 3]), idx: torch.Size([262144])
            # i: 2, neighborhood: torch.Size([128, 64, 8, 3]), center: torch.Size([128, 64, 3]), idx: torch.Size([65536])   idx -> B * G * M
            neighborhoods.append(neighborhood)
            centers.append(center)
            idxs.append(idx)  # neighbor indices
        # num pts 2048
        # 这个group多，和Point-MAE不太一样
        #   group_sizes: [16, 8, 8], 
        #   num_groups: [512, 256, 64],
        # exit(-1)

        # hierarchical encoder
        if eval:
            # for linear svm
            # raise NotImplementedError
            x_vis_list, _, _ = self.h_encoder(neighborhoods, centers, idxs, eval=True)
            x_vis = x_vis_list[-1]
            return x_vis.mean(1) + x_vis.max(1)[0]
        else:
            x_vis_list, x_mask_without_pos, x_mask_without_point = self.h_encoder(neighborhoods, centers, idxs)
            # , x_mask_no_pos_list, x_mask_no_point_list

        # multi scale patch matching (skip the decoder currently)
        x_mask_without_pos = self.pred_pos_proj(x_mask_without_pos) # B, Mask, C -> B, Mask, C
        x_mask_without_point = self.pred_point_proj(x_mask_without_point) # B, Mask, C -> B, Mask, C
        # # 可以试试合起来
        
        # x_mask_without_pos_before_norm = x_mask_without_pos
        # x_mask_without_point_before_norm = x_mask_without_point
        
        B, M, C = x_mask_without_pos.shape
        # B, M, C
        # InfoNCE loss between x_mask_without_pos and x_mask_without_point
        temperature = 0.5    # FIXME 这里是多少？ https://github.com/leftthomas/SimCLR/blob/master/main.py

        x_mask_without_pos = F.normalize(x_mask_without_pos, dim=-1)    # B, Mask, C
        x_mask_without_point = F.normalize(x_mask_without_point, dim=-1)    # B, Mask, C
        
        M = x_mask_without_point.shape[1]
        identity_mask = torch.eye(2 * M, device=x_mask_without_pos.device)  
        x_mask = torch.cat([x_mask_without_pos, x_mask_without_point], dim=1) # B, 2M, C
        sim_matrix = torch.bmm(x_mask, x_mask.transpose(1, 2)) / temperature  # B, 2M, 2M
        sim_matrix = torch.exp(sim_matrix)  # B, 2M, 2M
        sim_matrix = sim_matrix * (1 - identity_mask).unsqueeze(0).expand(B, -1, -1)    # B, 2M, 2M
        
        pos_sim = torch.exp(torch.sum(x_mask_without_pos * x_mask_without_point, dim=-1) / temperature)   # B, M
        pos_sim = torch.cat([pos_sim, pos_sim], dim=1)  # B, 2M
        loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()   # (B, 2M) / (B, 2M)
        # loss2 = torch.tensor(-1).cuda()        
        
        
        # hierarchical decoder
        # centers.reverse()
        # neighborhoods.reverse()
        # x_vis_list.reverse()
        # masks.reverse()

        # for i in range(len(self.decoder_dims)):
        #     center = centers[i]
        #     # 1st-layer decoder, concatenate visible and masked tokens
        #     if i == 0:
        #         x_full, mask = x_vis_list[i], masks[i]
        #         B, _, C = x_full.shape
        #         center_0 = torch.cat((center[~mask].reshape(B, -1, 3), center[mask].reshape(B, -1, 3)), dim=1)

        #         pos_emd_vis = self.decoder_pos_embeds[i](center[~mask]).reshape(B, -1, C)
        #         pos_emd_mask = self.decoder_pos_embeds[i](center[mask]).reshape(B, -1, C)
        #         pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

        #         _, N, _ = pos_emd_mask.shape
        #         mask_token = self.mask_token.expand(B, N, -1)
        #         x_full = torch.cat([x_full, mask_token], dim=1)
            
        #     else:
        #         x_vis = x_vis_list[i]
        #         bool_vis_pos = ~masks[i]
        #         mask_vis = mask_vis_list[i]
        #         B, N, _ = center.shape
        #         _, _, C = x_vis.shape
        #         x_full_en = torch.zeros(B, N, C).cuda()
        #         x_full_en[bool_vis_pos] = x_vis[mask_vis]       # x_vis skip connection

        #         # token propagation
        #         if i == 1:
        #             x_full = self.token_prop[i - 1](center, center_0, x_full_en, x_full)   # x_full
        #         else:
        #             x_full = self.token_prop[i - 1](center, centers[i - 1], x_full_en, x_full)
        #         pos_full = self.decoder_pos_embeds[i](center)

        #     x_full = self.h_decoder[i](x_full, pos_full)
        #     # print(f'i: {i}, x_full: {x_full.shape}')
        #     # i: 0, x_full: torch.Size([128, 64, 384])                                                                                                                                                               
        #     # i: 1, x_full: torch.Size([128, 256, 192])
        # # exit(-1)
        # # reconstruction      
        # x_full  = self.decoder_norm(x_full)
        # B, N, C = x_full.shape
        # x_rec = x_full[masks[-2]].reshape(-1, C)
        # L, _ = x_rec.shape

        # rec_points = self.rec_head(x_rec.unsqueeze(-1)).reshape(L, -1, 3)   # torch.Size([21072, 16, 3])
        # gt_points = neighborhoods[-2][masks[-2]].reshape(L, -1, 3)  # torch.Size([21072, 8, 3])
        
        # # exit(-1)

        # # CD loss
        # loss = self.rec_loss(rec_points, gt_points)
        return loss
