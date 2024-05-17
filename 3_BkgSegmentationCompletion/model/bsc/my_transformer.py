import math
from turtle import shape
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if "Linear" in classname or "Embedding" == classname:
        print(f"Initializing Module {classname}.")
        nn.init.trunc_normal_(m.weight.data, 0.0, 0.02)
    # elif "Parameter" in classname:
    #     return nn.init.trunc_normal_(m, 0.0, 0.02)

class Attention(nn.Module):
    """
    Simple Self-Attention algorithm. Potential for optimization using a non-quadratic attention mechanism in complexity.
    -> Linformer, Reformer etc.
    """
    def __init__(self, dim=768, heads=8):
        super(Attention, self).__init__()
        d = dim // heads
        self.q, self.k, self.v = nn.Linear(dim, d), nn.Linear(dim, d), nn.Linear(dim, d)
        self.norm = d ** 0.5
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        qk = torch.softmax(q @ torch.transpose(k, 1, 2) / self.norm, dim=1)
        qk = self.dropout(qk)
        attn = torch.matmul(qk, v)
        return attn


class MultiHeadAttention(nn.Module):
    """
    Implementation of MultiHeadAttention, splitting it up to multiple Self-Attention layers and concatenating
    the results and subsequently running it through one linear layer of same dimension.
    """
    def __init__(self, dim=768, heads=8):
        super(MultiHeadAttention, self).__init__()
        self.self_attention_heads = nn.ModuleList([Attention(dim, heads) for _ in range(heads)])
        self.projector = nn.Linear(dim, dim)

    def forward(self, x):
        for i, sa_head in enumerate(self.self_attention_heads):
            if i == 0:
                out = sa_head(x)
            else:
                out = torch.cat((out, sa_head(x)), axis=-1)
        out = self.projector(out)
        return out

# def window_partition(x, window_size):
#     """
#     Args:
#         x: (B, H, W, C)
#         window_size (int): window size
#     Returns:
#         windows: (num_windows*B, window_size, window_size, C)
#     """
#     B, H, W, C = x.shape
#     x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
#     windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
#     return windows

# def window_partition(x, window_size):
#     """
#     Args:
#         x: (B, C, H, W)
#         window_size (int): window size
#     Returns:
#         windows: (num_windows*B, window_size, window_size, C)
#     """
#     B, C, H, W = x.shape
#     x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
#     windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size, window_size, C)
#     return windows

def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    windows = rearrange(x, "B C (num_h win_h) (num_w win_w) -> (B num_h num_w) (win_h win_w) C", num_h=H//window_size, num_w=W//window_size)
    return windows

def window_merge(x, window_numb, window_size, batch_size):

    """
    Args:
        x: (B num_h num_w) (win_h win_w) C
        window_numb (int): window numb
        window_size (int): window size
        batch_size (int): batch size
    Returns:
        merged: (B, C, H, W)    
    """
    x = rearrange(x, "(B num_h num_w) (win_h win_w) C -> B C (num_h win_h) (num_w win_w)", B=batch_size, num_h=window_numb, win_h=window_size)
    return x

class basic_block(nn.Module):
    def __init__(self, dim=768, hidden_dim=3072, heads=8, window_size=8):
        super(basic_block, self).__init__()
        self.window_size = window_size
        self.local_msa = nn.MultiheadAttention(dim, num_heads=8, batch_first=True, dropout=0.1)
        self.global_msa = nn.MultiheadAttention(dim, num_heads=8, batch_first=True, dropout=0.1)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = nn.Sequential(*[
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        ])
        self.dropout = nn.Dropout(p=0.1)
        self.avgpool = nn.AvgPool2d(kernel_size=4)
        self.upsample = nn.Upsample(scale_factor=4, mode="bicubic")

    def forward(self, x, mask):
        x_original = x.clone()
        batch, _, height, width = x.shape

        ## global
        x_down = self.avgpool(x)
        x_down = rearrange(x_down, "B E H W -> B (H W) E")

        global_attn, _ = self.global_msa(x_down, x_down, x_down)
        global_attn = rearrange(global_attn, "B (H W) E -> B E H W", H=height//4)
        global_attn = self.upsample(global_attn)
        global_attn = self.dropout(global_attn)

        ## local
        x_windows = window_partition(x, window_size=self.window_size)
        local_attn, _ = self.local_msa(x_windows, x_windows, x_windows)
        local_attn = window_merge(local_attn, window_numb=height//self.window_size, window_size=self.window_size, batch_size=batch)
        local_attn = self.dropout(local_attn)

        x = x.add(global_attn)
        x = x.add(local_attn)

        x = rearrange(x, "B E H W -> B (H W) E")
        x = self.LayerNorm1(x)
        mlp = self.MLP(x)
        x = x.add(mlp)
        x = self.LayerNorm2(x)

        x = rearrange(x, "B (H W) E -> B E H W", H=height)
        #mask = repeat(mask, 'B H W -> B repeat H W', repeat=x.shape[1])
        #x[~mask] = x_original[~mask]
        return x 

class BidirectionalTransformer_local_global(nn.Module):
    def __init__(self, args):
        super(BidirectionalTransformer_local_global, self).__init__()
        self.args = args
        self.num_image_tokens = args['inter_seg_size']**2
        self.tok_emb = nn.Embedding(args['label_nc'] + 2, args['dim'])
        self.pos_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(args['inter_seg_size']*args['inter_seg_size'], args['dim'])), 0., 0.02)
        self.ln = nn.LayerNorm(args['dim'], eps=1e-12)
        self.drop = nn.Dropout(p=0.1)       
        self.blocks = nn.Sequential(*[basic_block(args['dim'], args['hidden_dim'], args['n_heads'], args['window_size']) for _ in range(args['n_layers'])])

        self.final_token_prediction = nn.Sequential(*[
            nn.Linear(in_features=args['dim'], out_features=args['dim']),
            nn.GELU(),
            nn.LayerNorm(args['dim'], eps=1e-12)
        ])

        self.final_bias = nn.Parameter(torch.zeros(args['inter_seg_size'], args['inter_seg_size'], args['label_nc'] + 2))
        self.apply(weights_init)

    def forward(self, x, mask):

        token_embeddings = self.tok_emb(x)
        token_embeddings = rearrange(token_embeddings, "B H W E-> B (H W) E")

        position_embeddings = self.pos_emb
        embed = self.drop(self.ln(token_embeddings + position_embeddings))
        embed = rearrange(embed, "B (H W) E-> B E H W", H=self.args['inter_seg_size'])

        for block in self.blocks:
            embed = block(embed, mask)
        #embed = self.blocks(embed, mask)

        embed = rearrange(embed, "B E H W -> B H W E", H=self.args['inter_seg_size'])

        final_embed = self.final_token_prediction(embed)
        final_logits = torch.matmul(final_embed, self.tok_emb.weight.T) + self.final_bias

        return None, final_logits

from model.bsc.swin_transformer import BasicLayer, PatchEmbed, PatchMerging

class BidirectionalTransformer_swin_mask(nn.Module):
    def __init__(self, args):
        super(BidirectionalTransformer_swin_mask, self).__init__()
        self.args = args
        self.num_image_tokens = args['inter_seg_size']**2
        self.tok_emb = nn.Embedding(args['label_nc'] + 2, args['dim'])
        self.pos_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(args['inter_seg_size']*args['inter_seg_size'], args['dim'])), 0., 0.02)
        self.ln = nn.LayerNorm(args['dim'], eps=1e-12)
        self.drop = nn.Dropout(p=0.1)       


        patches_resolution = (args['inter_seg_size'], args['inter_seg_size'])
        # build layers
        self.blocks = nn.ModuleList()
        self.swin_num_layers = 4
        depths = [2, 2, 6, 2]
        num_heads = [4, 4, 8, 16]
        for i_layer in range(self.swin_num_layers):
            layer = BasicLayer(dim=int(args['dim']),
                               input_resolution=(patches_resolution[0],
                                                 patches_resolution[1]),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=16,
                               mlp_ratio=4,
                               qkv_bias=True, qk_scale=None,
                               drop=0, attn_drop=0,
                               drop_path=0,
                               norm_layer=nn.LayerNorm,
                               downsample=None,
                               use_checkpoint=False)
            self.blocks.append(layer)


        self.final_token_prediction = nn.Sequential(*[
            nn.Linear(in_features=args['dim'], out_features=args['dim']),
            nn.GELU(),
            nn.LayerNorm(args['dim'], eps=1e-12)
        ])

        self.final_bias = nn.Parameter(torch.zeros(args['inter_seg_size'], args['inter_seg_size'], args['label_nc'] + 2))

        self.gamma = self.gamma_func("cosine")
        self.choice_temperature = args['temperature']
        self.mask_token_id = args['label_nc'] + 1

        self.no_multinomial = args['no_multinomial']
        
        self.apply(weights_init)

    def gamma_func(self, mode="cosine"):
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        elif mode == "cubic":
            return lambda r: 1 - r ** 3
        else:
            raise NotImplementedError

    def forward(self, x, mr_mask, missing_class):

        mr_mask = rearrange(mr_mask, "B H W -> B (H W)")
        sequence = rearrange(x, "B H W -> B (H W)")
        sequence_length = sequence.shape[1]

        r = math.floor(self.gamma(np.random.uniform()) * sequence_length)
        topk_indices = torch.rand(sequence.shape).topk(r, dim=1).indices.type_as(sequence)
        mask = torch.zeros(sequence.shape).type_as(sequence).bool()
        mask.scatter_(dim=1, index=topk_indices, value=True)
        total_mask = torch.logical_or(~mask, ~mr_mask)
        mask_token_id = (torch.ones_like(missing_class)*201).unsqueeze(1)
        masked_indices = mask_token_id*torch.ones_like(sequence).type_as(sequence).long()
        input_sequence = total_mask * sequence + (~total_mask) * masked_indices

        token_embeddings = self.tok_emb(input_sequence)
        #token_embeddings = rearrange(token_embeddings, "B H W E -> B (H W) E")

        position_embeddings = self.pos_emb
        embed = self.drop(self.ln(token_embeddings + position_embeddings))
        #embed = rearrange(embed, "B (H W) E-> B E H W", H=self.args['inter_seg_size'])

        for block in self.blocks:
            embed = block(embed)
        #embed = self.blocks(embed, mask)

        embed = rearrange(embed, "B (H W) E -> B H W E", H=self.args['inter_seg_size'])

        final_embed = self.final_token_prediction(embed)
        final_logits = torch.matmul(final_embed, self.tok_emb.weight.T[:,:202]) + self.final_bias

        input_seg = rearrange(input_sequence, "B (H W) -> B H W", H=self.args['inter_seg_size'])
        return None, final_logits, input_seg
    
    def predict_logits(self, input_sequence):
        token_embeddings = self.tok_emb(input_sequence)
        #token_embeddings = rearrange(token_embeddings, "B H W E -> B (H W) E")

        position_embeddings = self.pos_emb
        embed = self.drop(self.ln(token_embeddings + position_embeddings))
        #embed = rearrange(embed, "B (H W) E-> B E H W", H=self.args['inter_seg_size'])

        for block in self.blocks:
            embed = block(embed)
        #embed = self.blocks(embed, mask)

        embed = rearrange(embed, "B (H W) E -> B H W E", H=self.args['inter_seg_size'])

        final_embed = self.final_token_prediction(embed)
        final_logits = torch.matmul(final_embed, self.tok_emb.weight.T[:,:202]) + self.final_bias
        final_logits = rearrange(final_logits, "B H W class -> B (H W) class", H=self.args['inter_seg_size'])

        return final_logits

    def mask_by_random_topk(self, mask_len, probs, temperature=1.0):
        confidence = torch.log(probs) + temperature * torch.distributions.gumbel.Gumbel(0, 1).sample(probs.shape).to("cuda")
        sorted_confidence, _ = torch.sort(confidence, dim=-1)
        # Obtains cut off threshold given the mask lengths.
        cut_off = torch.take_along_dim(sorted_confidence, mask_len.to(torch.long), dim=-1)
        # Masks tokens with lower confidence.
        masking = (confidence < cut_off)
        return masking

    @torch.no_grad()
    def revise_phase(self, x, mr_mask, missing_class, T=8):
        cur_revised_sequence_per_step = []

        mr_mask = rearrange(mr_mask, "B H W -> B (H W)")
        cur_sequence = rearrange(x, "B H W -> B (H W)")
        sequence_length = cur_sequence.shape[1]

        for t in range(T):
            r = math.floor(self.gamma(np.random.uniform(low=0.5, high=1)) * sequence_length)
            topk_indices = torch.rand(cur_sequence.shape).topk(r, dim=1).indices.type_as(cur_sequence)
            mask = torch.zeros(cur_sequence.shape).type_as(cur_sequence).bool()
            mask.scatter_(dim=1, index=topk_indices, value=True)
            total_mask = torch.logical_or(~mask, ~mr_mask)
            mask_token_id = (torch.ones_like(missing_class)*201).unsqueeze(1)
            masked_indices = mask_token_id*torch.ones_like(cur_sequence).type_as(cur_sequence).long()
            cur_sequence = total_mask * cur_sequence + (~total_mask) * masked_indices

            logits = self.predict_logits(cur_sequence)
            _, sampled_ids = torch.max(logits[:,:,:-1], dim=-1)
            unknown_map = (cur_sequence == mask_token_id)  # which tokens need to be sampled -> bool [8, 257]
            cur_sequence = torch.where(unknown_map, sampled_ids, cur_sequence)  # replace all -1 with their samples and leave the others untouched [8, 257]
            cur_revised_sequence_per_step.append(rearrange(cur_sequence, "B (H W) -> B H W", H=self.args['inter_seg_size']))
        
        return cur_revised_sequence_per_step

    @torch.no_grad()
    def iterative_decoding(self, x, mr_mask, missing_class, T=16):
        cur_sequence_per_step = []

        mr_mask = rearrange(mr_mask, "B H W -> B (H W)")
        total_mask = ~mr_mask
        sequence = rearrange(x, "B H W -> B (H W)")

        mask_token_id = (missing_class).unsqueeze(1)
        masked_indices = mask_token_id*torch.ones_like(sequence).type_as(sequence).long()
        cur_sequence = total_mask * sequence + (~total_mask) * masked_indices
        unknown_number_in_the_beginning = torch.sum(cur_sequence == mask_token_id, dim=-1)

        for t in range(T):
            logits = self.predict_logits(cur_sequence)
            
            if self.no_multinomial:
                _, sampled_ids = torch.max(logits[:,:,:-1], dim=-1)
            else:  
                sampled_ids = torch.distributions.categorical.Categorical(logits=logits[:,:,:-1]).sample()

            unknown_map = (cur_sequence == mask_token_id)  # which tokens need to be sampled -> bool [8, 257]
            sampled_ids = torch.where(unknown_map, sampled_ids, cur_sequence)  # replace all -1 with their samples and leave the others untouched [8, 257]

            ratio = 1. * (t + 1) / T  # just a percentage e.g. 1 / 12
            mask_ratio = self.gamma(ratio)

            probs = F.softmax(logits, dim=-1)  # convert logits into probs [8, 257, 1024]
            selected_probs = torch.squeeze(torch.take_along_dim(probs, torch.unsqueeze(sampled_ids, -1), -1), -1)  # get probability for selected tokens in categorical call, also for already sampled ones [8, 257]
            selected_probs = torch.where(unknown_map, selected_probs, torch.Tensor([torch.inf]).type_as(logits))  # ignore tokens which are already sampled [8, 257]

            mask_len = torch.unsqueeze(torch.floor(unknown_number_in_the_beginning * mask_ratio), 1)  # floor(256 * 0.99) = 254 --> [254, 254, 254, 254, ....]
            mask_len = torch.maximum(torch.zeros_like(mask_len), torch.minimum(torch.sum(unknown_map, dim=-1, keepdim=True)-1, mask_len))  # add -1 later when conditioning and also ones_like. Zeroes just because we have no cond token
            # max(1, min(how many unknown tokens, how many tokens we want to sample))

            # Adds noise for randomness
            masking = self.mask_by_random_topk(mask_len, selected_probs, temperature=self.choice_temperature * (1. - ratio))
            # Masks tokens with lower confidence.
            cur_sequence = torch.where(masking, mask_token_id, sampled_ids)

            cur_sequence_per_step.append(rearrange(cur_sequence, "B (H W) -> B H W", H=self.args['inter_seg_size']))

        return cur_sequence_per_step



class BidirectionalTransformer_local_global_mask(nn.Module):
    def __init__(self, args):
        super(BidirectionalTransformer_local_global_mask, self).__init__()
        self.args = args
        self.num_image_tokens = args['inter_seg_size']**2
        self.tok_emb = nn.Embedding(args['label_nc'] + 2, args['dim'])
        self.pos_emb = nn.init.trunc_normal_(nn.Parameter(torch.zeros(args['inter_seg_size']*args['inter_seg_size'], args['dim'])), 0., 0.02)
        self.ln = nn.LayerNorm(args['dim'], eps=1e-12)
        self.drop = nn.Dropout(p=0.1)       
        self.blocks = nn.Sequential(*[basic_block(args['dim'], args['hidden_dim'], args['n_heads'], args['window_size']) for _ in range(args['n_layers'])])

        self.final_token_prediction = nn.Sequential(*[
            nn.Linear(in_features=args['dim'], out_features=args['dim']),
            nn.GELU(),
            nn.LayerNorm(args['dim'], eps=1e-12)
        ])

        self.final_bias = nn.Parameter(torch.zeros(args['inter_seg_size'], args['inter_seg_size'], args['label_nc'] + 2))

        self.gamma = self.gamma_func("cosine")
        self.choice_temperature = 4.5
        self.mask_token_id = args['label_nc'] + 1

        self.apply(weights_init)

    def gamma_func(self, mode="cosine"):
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        elif mode == "cubic":
            return lambda r: 1 - r ** 3
        else:
            raise NotImplementedError

    def forward(self, x, mr_mask):

        mr_mask = rearrange(mr_mask, "B H W -> B (H W)")
        sequence = rearrange(x, "B H W -> B (H W)")
        sequence_length = sequence.shape[1]

        r = math.floor(self.gamma(np.random.uniform()) * sequence_length)
        topk_indices = torch.rand(sequence.shape).topk(r, dim=1).indices.type_as(sequence)
        mask = torch.zeros(sequence.shape).type_as(sequence).bool()
        mask.scatter_(dim=1, index=topk_indices, value=True)
        total_mask = torch.logical_or(~mask, ~mr_mask)

        masked_indices = self.mask_token_id*torch.ones_like(sequence).type_as(sequence).long()
        
        input_sequence = total_mask * sequence + (~total_mask) * masked_indices

        token_embeddings = self.tok_emb(input_sequence)
        #token_embeddings = rearrange(token_embeddings, "B H W E -> B (H W) E")

        position_embeddings = self.pos_emb
        embed = self.drop(self.ln(token_embeddings + position_embeddings))
        embed = rearrange(embed, "B (H W) E-> B E H W", H=self.args['inter_seg_size'])

        for block in self.blocks:
            embed = block(embed, mr_mask)
        #embed = self.blocks(embed, mask)

        embed = rearrange(embed, "B E H W -> B H W E", H=self.args['inter_seg_size'])

        final_embed = self.final_token_prediction(embed)
        final_logits = torch.matmul(final_embed, self.tok_emb.weight.T) + self.final_bias

        input_seg = rearrange(input_sequence, "B (H W) -> B H W", H=self.args['inter_seg_size'])
        return None, final_logits, input_seg
    
    def predict_logits(self, input_sequence):
        token_embeddings = self.tok_emb(input_sequence)
        #token_embeddings = rearrange(token_embeddings, "B H W E -> B (H W) E")

        position_embeddings = self.pos_emb
        embed = self.drop(self.ln(token_embeddings + position_embeddings))
        embed = rearrange(embed, "B (H W) E-> B E H W", H=self.args['inter_seg_size'])

        for block in self.blocks:
            embed = block(embed, None)
        #embed = self.blocks(embed, mask)

        embed = rearrange(embed, "B E H W -> B H W E", H=self.args['inter_seg_size'])

        final_embed = self.final_token_prediction(embed)
        final_logits = torch.matmul(final_embed, self.tok_emb.weight.T) + self.final_bias
        final_logits = rearrange(final_logits, "B H W class -> B (H W) class", H=self.args['inter_seg_size'])

        return final_logits

    def mask_by_random_topk(self, mask_len, probs, temperature=1.0):
        confidence = torch.log(probs) + temperature * torch.distributions.gumbel.Gumbel(0, 1).sample(probs.shape).to("cuda")
        sorted_confidence, _ = torch.sort(confidence, dim=-1)
        # Obtains cut off threshold given the mask lengths.
        cut_off = torch.take_along_dim(sorted_confidence, mask_len.to(torch.long), dim=-1)
        # Masks tokens with lower confidence.
        masking = (confidence < cut_off)
        return masking

    @torch.no_grad()
    def iterative_decoding(self, x, mr_mask, T=16):
        cur_sequence_per_step = []

        mr_mask = rearrange(mr_mask, "B H W -> B (H W)")
        total_mask = ~mr_mask
        sequence = rearrange(x, "B H W -> B (H W)")
        masked_indices = self.mask_token_id*torch.ones_like(sequence).type_as(sequence).long()
        cur_sequence = total_mask * sequence + (~total_mask) * masked_indices
        unknown_number_in_the_beginning = torch.sum(cur_sequence == self.mask_token_id, dim=-1)

        for t in range(T):
            logits = self.predict_logits(cur_sequence)
            sampled_ids = torch.distributions.categorical.Categorical(logits=logits).sample()

            unknown_map = (cur_sequence == self.mask_token_id)  # which tokens need to be sampled -> bool [8, 257]
            sampled_ids = torch.where(unknown_map, sampled_ids, cur_sequence)  # replace all -1 with their samples and leave the others untouched [8, 257]

            ratio = 1. * (t + 1) / T  # just a percentage e.g. 1 / 12
            mask_ratio = self.gamma(ratio)

            probs = F.softmax(logits, dim=-1)  # convert logits into probs [8, 257, 1024]
            selected_probs = torch.squeeze(torch.take_along_dim(probs, torch.unsqueeze(sampled_ids, -1), -1), -1)  # get probability for selected tokens in categorical call, also for already sampled ones [8, 257]
            selected_probs = torch.where(unknown_map, selected_probs, torch.Tensor([torch.inf]).type_as(logits))  # ignore tokens which are already sampled [8, 257]

            mask_len = torch.unsqueeze(torch.floor(unknown_number_in_the_beginning * mask_ratio), 1)  # floor(256 * 0.99) = 254 --> [254, 254, 254, 254, ....]
            mask_len = torch.maximum(torch.zeros_like(mask_len), torch.minimum(torch.sum(unknown_map, dim=-1, keepdim=True)-1, mask_len))  # add -1 later when conditioning and also ones_like. Zeroes just because we have no cond token
            # max(1, min(how many unknown tokens, how many tokens we want to sample))

            # Adds noise for randomness
            masking = self.mask_by_random_topk(mask_len, selected_probs, temperature=self.choice_temperature * (1. - ratio))
            # Masks tokens with lower confidence.
            cur_sequence = torch.where(masking, self.mask_token_id, sampled_ids)

            cur_sequence_per_step.append(rearrange(cur_sequence, "B (H W) -> B H W", H=self.args['inter_seg_size']))

        return cur_sequence_per_step
