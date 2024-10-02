"""
Single-file implementation of autoregressive policy. It requires pytorch and timm (`pip install timm torch`).

Running this file will trains a simple chunking causal transformer that generates Binary MNIST images.

```
python arp.py
```

Generated images are saved in `mnist_generated_arp` folder.
"""

import torch, math, random
from collections.abc import Iterable
from collections import defaultdict
from typing import List, Tuple, TypedDict, Union, Dict, Optional, Callable, FrozenSet, Any, TypeVar, Literal
import itertools
from copy import deepcopy
from torch import Tensor, nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp
import numpy as np
import torch.distributions as D
##

#region Chunk Transformer Layer

def modulate(x, shift, scale):
    """ x: (bs, L, d)
        shift: (bs, L, d)
        scale: (bs, L, d)
    """
    return x * (1 + scale) + shift


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

def clamp_dtype_min_max(v, dtype, inplace=True):
    MIN, MAX = torch.finfo(dtype).min, torch.finfo(dtype).max
    return v.clamp_(MIN, MAX) if inplace else v.clamp(MIN, MAX)

class Attention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop=0.1, resid_pdrop=0.1, cross=False, clamp_attn=False):
        super().__init__()
        self.clamp_attn = clamp_attn
        assert n_embd % n_head == 0
        self.cross = cross
        if cross:
            self.kv_attn = nn.Linear(n_embd, 2 * n_embd)
            self.q_attn = nn.Linear(n_embd, n_embd)
        else:
            self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.n_head = n_head
        self.n_embd = n_embd
    
    def attend(self, q, k):
        attn = q @ (k.transpose(-2, -1) / math.sqrt(k.size(-1)))
        dtype = attn.dtype
        if self.clamp_attn and dtype == torch.float16:
            return clamp_dtype_min_max(attn, dtype)
        else:
            return attn
    
    def forward_interleave(self, xs, attn_masks, dependency_attn_mask=None):
        assert not self.cross 
        B, T, C = xs[0].size() 
        dev = xs[0].device
        qkvs = []
        for x in xs:
            q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            qkvs.append([q,k,v])
        
        (q_star, k_star, v_star), (q_hat, k_hat, v_hat) = qkvs
        (mask_star, mask_hat), mask_causal = [~m for m in attn_masks], torch.tril(torch.ones(T, T, device=dev))[None, None, ...] == 0

        def mlp(y):
            y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
            y = self.resid_dropout(self.c_proj(y))
            return y
        
        def apply_mask(att, mask, val=float('-inf')):
            if len(mask.shape) == 3: mask = mask[:, None, :, :]
            return att.masked_fill(mask, val)
        
        def merge_attn_logits(att_star, att_hat):
            valid_pos = ~torch.isinf(att_hat)
            att_star = att_star.clone()
            att_star[valid_pos] = att_hat[valid_pos]
            return att_star
            
        softmax = nn.Softmax(dim=-1)

        att_star = self.attend(q_star, k_star)
        if dependency_attn_mask is not None:
            att_star = apply_mask(att_star, ~dependency_attn_mask)

        y_star = mlp(self.attn_dropout(softmax(apply_mask(att_star, mask_causal))) @ v_star)

        attn_hat = merge_attn_logits(
            apply_mask(self.attend(q_hat, k_star) , mask_star),
            apply_mask(self.attend(q_hat, k_hat) , mask_hat)
        )
        if dependency_attn_mask is not None: attn_hat = apply_mask(attn_hat, ~dependency_attn_mask)
        attn_hat = self.attn_dropout(softmax(attn_hat))

        y_hat = apply_mask(attn_hat, mask_star, 0) @ v_star + apply_mask(attn_hat, mask_hat, 0) @ v_hat
        y_hat = mlp(y_hat)
        return y_star, y_hat


    def forward(self, x, c=None, attn_mask=None):
        """
        x: (B, T, C) input sequence
        c: (B, L, C) context sequence
        attn_mask: (B | 1, T, T) attention mask (True means keep, False means blocking)
        """
        B, T, C = x.size() 
        if self.cross:
            k, v = self.kv_attn(c).split(self.n_embd, dim=2)
            q = self.q_attn(x)
            Tc = c.size(1)
            k = k.view(B, Tc, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = v.view(B, Tc, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        else:
            q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        att = self.attend(q, k)
        if attn_mask is not None:
            attn_mask = attn_mask[:, None, :, :]
            att = att.masked_fill(~attn_mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.resid_dropout(self.c_proj(y))
        return y


class ChunkTransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, mlp_dropout=0.1,
                 attn_kwargs={}, cond_attn_kwargs={},
                 conditional=False, AdaLN=False, norm_before_AdaLN=False):
        super().__init__()
        self.ln_attn = nn.LayerNorm(hidden_size, elementwise_affine=not AdaLN, eps=1e-6)
        self.ln_mlp = nn.LayerNorm(hidden_size, elementwise_affine=not AdaLN, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, 
                       drop=mlp_dropout)
        self.attn = Attention(hidden_size, num_heads, **attn_kwargs)
        
        self.conditional = conditional
        self.AdaLN = AdaLN
        if conditional:
            self.norm_cond = nn.LayerNorm(hidden_size, eps=1e-6)
            self.cond_attn = Attention(hidden_size, num_heads, **cond_attn_kwargs, cross=True)
            if AdaLN:
                self.adaLN_modulation = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(hidden_size, 6 * hidden_size, bias=True)
                )
                self.norm_before_AdaLN = norm_before_AdaLN
                if norm_before_AdaLN:
                    self.ln_ada = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    
    # mask will be inversed before used with mask_filled
    # therefore, false -> blocking
    
    @staticmethod
    def train_attn_masks(chunk_ids):
        """
        chunk_ids: (B | 1, L) chunk ids, starting from 0, and ordered
        """
        m_star = chunk_ids[:, :, None] > chunk_ids[:, None, :]
        m_hat = chunk_ids[:, :, None] == chunk_ids[:, None, :]
        return m_star, m_hat

    @staticmethod
    def eval_attn_mask(chunk_ids):
        L = chunk_ids.size(1)
        prompt = chunk_ids[:, -1:] # (B, 1)
        m = (chunk_ids[:, :, None] == prompt[:, None, :]).repeat(1, 1, L)
        m = m | (torch.tril(torch.ones(L, L, device=chunk_ids.device))[None, ...] == 1)
        return m
    
    @staticmethod
    def dependency_attn_mask(tk_types, block_attn_directions):
        """
        tk_types: (B, L) 
        block_attn_directions: list of (int, int), where each tuple is (curr, other) to block attention
        """
        bs, L = tk_types.shape
        mask = torch.full([bs, L, L], fill_value=True, device=tk_types.device, dtype=torch.bool)
        for from_, to in block_attn_directions:
            mask_ = (tk_types[:, :, None] == from_) & (tk_types[:, None, :] == to)
            mask = mask & (~mask_)
        return mask

    def forward_train(self, xs, c, masks, dependency_attn_mask=None):
        """
        xs: [(B, T, C), (B, T, C)] 
        masks: [(B | 1, T, T), (B | 1, T, T)]
        """
        is_conditional = self.conditional and c is not None
        cond_attns = [self.cond_attn(self.norm_cond(x), c) if is_conditional else 0 for x in xs]
        if self.AdaLN and is_conditional:
            if self.norm_before_AdaLN:  cond_attns = [self.ln_ada(cond_attn) for cond_attn in cond_attns]
            gates = [self.adaLN_modulation(cond_attn).chunk(6, dim=-1) for cond_attn in cond_attns]
            ys = self.attn.forward_interleave([modulate(self.ln_attn(x), shift_msa, scale_msa)
                  for x, (shift_msa, scale_msa, _, _, _, _) in zip(xs, gates)], masks, dependency_attn_mask=dependency_attn_mask)
            xs = [x + gate_msa * y for x, y, (_, _, gate_msa, _, _, _) in zip(xs, ys, gates)]
            xs = [x + gate_mlp * self.mlp(modulate(self.ln_mlp(x), shift_mlp, scale_mlp))
                  for x, (_, _, _, shift_mlp, scale_mlp, gate_mlp) in zip(xs, gates)]
        else:
            xs = [x + cond_attn for x, cond_attn in zip(xs, cond_attns)]
            ys = self.attn.forward_interleave([self.ln_attn(x) for x in xs], masks, dependency_attn_mask=dependency_attn_mask)
            xs = [x + y for x, y in zip(xs, ys)]
            xs = [x + self.mlp(self.ln_mlp(x)) for x in xs]
        return xs
    
    def forward_inference(self, x, c, mask=None):
        """
        x: (B, T, C) input sequence
        c: (B, L, C) context sequence

        """
        is_conditional = self.conditional and c is not None
        cond_attn = self.cond_attn(self.norm_cond(x), c) if is_conditional else 0
        if self.AdaLN and is_conditional:
            if self.norm_before_AdaLN: cond_attn = self.ln_ada(cond_attn)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond_attn).chunk(6, dim=-1)
            x = x + gate_msa * self.attn(modulate(self.ln_attn(x), shift_msa, scale_msa), attn_mask=mask)
            x = x + gate_mlp * self.mlp(modulate(self.ln_mlp(x), shift_mlp, scale_mlp))
        else:
            x = x + cond_attn
            x = x + self.attn(self.ln_attn(x), attn_mask=mask)
            x = x + self.mlp(self.ln_mlp(x))
        return x

#endregion #######################


#region Interface & Utilities

class TokenType(TypedDict):
    id: int
    name: str
    is_control: bool
    dim: int # number of dimensions

    # NOTE: different tokens may need to be embedded or predicted differently
    embedding: str # "discrete", "linear"
    predictor: str

    embedding_kwargs: Dict
    predictor_kwargs: Dict

    # NOTE: use for encoding
    is_continuous: bool
    bounds: List[float] # [min_of_dim1, max_of_dim1, min_of_dim2, max...] 
    dict_sizes: List[int] # [dict_size_of_dim1, dict_size_of_dim2, ...]

    @staticmethod
    def make(**kwargs):
        return {**{
            'is_control': False,
            'name': f'token-{random.randint(0, 1000)}',
            'dim': 1,
            'is_continuous': False,
            'dict_sizes': [1],
            'embedding_kwargs': {},
            'predictor_kwargs': {},
            'embedding': 'discrete',
            'predictor': 'class'
        }, **kwargs}
        

class LayerType(TypedDict):
    name: str 
    n_head: int 
    attn_kwargs: dict
    cond_attn_kwargs: dict
    mlp_dropout: float
    mlp_ratio: float

    AdaLN: Union[bool, str]
    norm_before_AdaLN: bool

    condition_on: str = ""

    @staticmethod
    def make(**kwargs):
        return {
            'name': "",
            'n_head': 4,
            'attn_kwargs': dict(attn_pdrop=0.1, resid_pdrop=0.1),
            'cond_attn_kwargs': dict(attn_pdrop=0.1, resid_pdrop=0.1),
            'mlp_dropout': 0.1,
            'mlp_ratio': 4.0,
            'norm_before_AdaLN': False,
            'AdaLN': False,
            'condition_on': "",
            **kwargs
        }

class ModelConfig:
    def __init__(self, n_embd: int = 64, embd_pdrop: float = 0.1, max_seq_len: int = 1024, layer_norm_every_block: bool = True,
                max_chunk_size:int = 1, layers: List[LayerType] = [], tokens: List[TokenType] = [], **kwargs):
        self.n_embd: int = n_embd
        self.embd_pdrop: float = embd_pdrop 
        self.max_chunk_size: int = max_chunk_size  
        self.layer_norm_every_block: bool = layer_norm_every_block
        self.max_seq_len: int = max_seq_len
        self.layers: List[LayerType] = layers
        self.tokens: List[TokenType] = tokens
        for k, v in kwargs.items():
            setattr(self, k, v)
        for i, token in enumerate(self.tokens):
            token['id'] = i

class IncompleteToken(TypedDict):
    chk_id: int 
    tk_id: int
    tk_val: int

def _make_registry():
    def chunk(name):
        def func(cls):
            chunk.map[name] = cls
            cls.name = name 
            return cls
        return func
    chunk.map = {}
    return chunk

register_token_embedding = _make_registry()
register_token_predictor = _make_registry()

T = TypeVar('T')
PerChunk = Union[Dict[Union[int, FrozenSet[int], Literal['default']], T], T]
SampleFunctionT = Callable[[List[Union[Tensor, D.Distribution]]], Tensor]
AttnDirectionsType = List[Union[Tuple[str, str], Tuple[int, int]]]
#endregion Interface ###############


#region Utility ###############


def flatten_per_chunk_dict(dct):
    return {k: v for ks, v in dct.items() for k in (ks if isinstance(ks, Iterable) else [ks])}


def pad_last_dim(tensor, target_size, val=0):
    if tensor.size(-1) < target_size:
        out = torch.full([*tensor.shape[:-1], target_size], fill_value=val, device=tensor.device, dtype=tensor.dtype)
        out[:, :, :tensor.size(-1)] = tensor
        return out
    else:
        return tensor

def cat_uneven_blc_tensors(*tensors):
    max_dim = max([t.size(-1) for t in tensors])
    return torch.cat([pad_last_dim(t, max_dim) for t in tensors], dim=1)

map2 = lambda func, nested_list: list(map(lambda sublist: list(map(func, sublist)), nested_list))

#endregion ####################


#region TokenCoder

class TokenCoder(nn.Module):
    def __init__(self, tokens: List[TokenType]):
        super().__init__()
        self.tokens: List[TokenType] = tokens
    
    def encode(self, tks, tk_ids):
        """ 
        tks: [*, dim], e.g., [B, T, dim]
        tk_ids: [*]
        return: [..., dim]
        """
        tks = tks.float().clone()
        tks_shape = tks.shape 
        for i, token in enumerate(self.tokens):
            mask = tk_ids == i
            tks[mask] = self.encode_ith(tks[mask], i, inplace=True).float()
        return tks.reshape(*tks_shape)
    
    def decode(self, tk_codes, tk_ids):
        """
        tk_codes: [*, dim] long
        tk_ids: [*]
        return: [*, dim] float
        """
        tk_codes = tk_codes.float()
        tks_shape = tk_codes.shape 
        for i, token in enumerate(self.tokens):
            mask = tk_ids == i
            tk_codes[mask] = self.decode_ith(tk_codes[mask], i, inplace=True)
        return tk_codes.reshape(*tks_shape)
    
    def need_encoding(self, token: TokenType, is_continuous: bool = None):
        if is_continuous is None:
            is_continuous = token['is_continuous']
        return is_continuous and register_token_embedding.map[token['embedding']].NEED_ENCODED_INPUT
    
    def encode_ith(self, tks: Tensor, i: int, inplace=False):
        """
        tks: [*, d], where d is the dimension of the i-th token type
        return: [*, d]
        """
        token = self.tokens[i]
        if self.need_encoding(token):
            out = torch.zeros_like(tks)
            tks = tks[..., :token['dim']]
            if not inplace: tks = tks.clone()
            for j in range(token['dim']):
                start, end = token['bounds'][2*j], token['bounds'][2*j+1]
                tks[..., j].clamp_(start, end)
                tks[..., j] -= start
                resolution = (end - start) / (token['dict_sizes'][j] - 1)
                tks[..., j] /= resolution
                tks[..., j].round_()
            out[..., :token['dim']] = tks  
            return out
        else:
            return tks
        
    def decode_ith(self, tk_codes: Tensor, i: int, inplace=False):
        token = self.tokens[i]
        if self.need_encoding(token):
            out = torch.zeros_like(tk_codes)
            tk_codes = tk_codes[..., :token['dim']]
            if not inplace: tk_codes = tk_codes.clone()
            for j in range(token['dim']):
                start, end = token['bounds'][2*j], token['bounds'][2*j+1]
                resolution = (end - start) / (token['dict_sizes'][j] - 1)
                tk_codes[..., j] = tk_codes[..., j].float() * resolution + start
            out[..., :token['dim']] = tk_codes
            return out
        else:
            return tk_codes

#endregion TokenCoder ###############


#region Embedding 

class TokenEmbeddingInterface(nn.Module):
    NEED_ENCODED_INPUT = False

    def __init__(self, n_embd: int, token: TokenType, **kwargs):
        super().__init__()
        self.n_embd = n_embd
        self.token = token
    

@register_token_embedding('zero')
class ZeroEmbedding(TokenEmbeddingInterface):
    def forward(self, tk_codes, **extra_contexts):
        return torch.zeros(*tk_codes.shape[:-1], self.n_embd, device=tk_codes.device)


@register_token_embedding('discrete')
class DiscreteEmbedding(TokenEmbeddingInterface):
    NEED_ENCODED_INPUT = True
    def __init__(self, n_embd: int, token: TokenType, embed_from: Optional[str]=None, **kwargs):
        super().__init__(n_embd, token)
        self.embed_from = embed_from
        if embed_from:
            assert token['dim'] == 1
        else:
            self.embed = nn.ModuleList([nn.Embedding(token['dict_sizes'][i], n_embd) for i in range(token['dim'])])
    
    def forward(self, tk_codes, **extra_contexts):
        if self.embed_from:
            weight = extra_contexts[self.embed_from]
            if weight.dim() == 2:
                out = F.embedding(tk_codes[..., 0].long(), weight)
            else:
                assert weight.dim() == 3
                bs = len(weight)
                tk_codes = tk_codes[..., 0].long().reshape(bs, -1, 1).repeat(1, 1, weight.size(-1))
                out = torch.gather(weight, 1, tk_codes).reshape(-1, self.n_embd)
        else:
            out = 0
            for j in range(self.token['dim']):
                out = out + self.embed[j](tk_codes[..., j].long())
        return out
    
    
@register_token_embedding('position_1d')
class Position1DEmbedding(TokenEmbeddingInterface):
    NEED_ENCODED_INPUT = False
    def __init__(self, n_embd: int, token: TokenType, scale=1.0, N=10000, **kwargs):
        super().__init__(n_embd, token)
        assert token['dim'] == 1
        self.scale = scale
        self.register_buffer("div_term", torch.exp(torch.arange(0, n_embd, 2) * (-math.log(N) / n_embd))[None, :])
        
    def forward(self, tk_codes, **extra_contexts):
        tk_codes = tk_codes[:, :self.token['dim']]
        x = torch.cat((
            torch.sin(self.scale * tk_codes * self.div_term),
            torch.cos(self.scale * tk_codes * self.div_term)), dim=1)
        x = x.view(-1, self.n_embd)
        return x

    
@register_token_embedding('position_2d')
class Position2DEmbedding(TokenEmbeddingInterface):
    NEED_ENCODED_INPUT = False
    def __init__(self, n_embd: int, token: TokenType, scale=1.0, N=10000, **kwargs):
        super().__init__(n_embd, token)
        assert token['dim'] == 2
        assert n_embd % 4 == 0
        self.scale = scale
        n_embd = n_embd // 2
        self.register_buffer("div_term", torch.exp(torch.arange(0, n_embd, 2) * (-math.log(N) / n_embd))[None, :])
        
    def forward(self, tk_codes, **extra_contexts):
        tk_codes = tk_codes[:, :self.token['dim']]
        pe = torch.zeros(tk_codes.size(0), self.n_embd, device=tk_codes.device) 
        d_model = self.n_embd // 2 
        pe[:, 0:d_model:2] = torch.sin(self.scale * tk_codes[:, :1] * self.div_term)
        pe[:, 1:d_model:2] = torch.cos(self.scale * tk_codes[:, :1] * self.div_term)       
        pe[:, d_model: :2] = torch.sin(self.scale * tk_codes[:, 1:] * self.div_term)
        pe[:, d_model: :2] = torch.cos(self.scale * tk_codes[:, 1:] * self.div_term)       
        return pe


@register_token_embedding('linear')
class LinearEmbedding(TokenEmbeddingInterface):
    NEED_ENCODED_INPUT = False
    def __init__(self, n_embd: int, token: TokenType, **kwargs):
        super().__init__(n_embd, token)
        self.embed = nn.Linear(token['dim'], n_embd)
    
    def forward(self, tk_codes, **extra_contexts):
        return self.embed(tk_codes[..., :self.token['dim']])


@register_token_embedding('feat_grid_2d')
class FeatureGrid2DEmbedding(TokenEmbeddingInterface):
    NEED_ENCODED_INPUT = False
    def __init__(self, n_embd: int, token: TokenType, sampling_from: str, stride: Union[int, Tuple[int, int]] = 1, 
                token_format='xy', **kwargs):
        super().__init__(n_embd, token)
        self.sampling_from = sampling_from
        self.token_format = token_format
        assert self.token['dim'] == 2
        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride

    def forward(self, tk_codes, **extra_contexts):
        """
        tk_codes: (*, 2)
        """
        assert self.sampling_from in extra_contexts, f"extra context {self.sampling_from} not found"
        feat_grid = extra_contexts[self.sampling_from]
        grid_shape = feat_grid.shape # (B, C, H, W)
        tk_codes = tk_codes[..., :self.token['dim']].clone().float()
        tk_codes[..., 0] /= self.stride[0]
        tk_codes[..., 1] /= self.stride[1]
        tk_codes_shape = tk_codes.shape
        tk_codes = tk_codes.reshape(grid_shape[0], -1, 2)
        embs = self.grid_sample(tk_codes, feat_grid)
        return embs.reshape(*tk_codes_shape[:-1], self.n_embd)
    
    @staticmethod
    def batched_index_select(inp, dim, index):
        """
        input: B x * x ... x *
        dim: scalar > 0 (not batch dim)
        index: B x M
        """
        views = [inp.shape[0]] + [1 if i != dim else -1 for i in range(1, len(inp.shape))]
        expanse = list(inp.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(inp, dim, index)

    def grid_sample(
        self, points: torch.Tensor, feat_grid: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """
        :param points: [B, P, 2], where P is the number of points
        :param feat_grid: size [B, C, H, W]
        :return: the weighted average for each point according to the hm values. the size is [nc, npt, 1]. 
        """
        nc, nw, h, w = feat_grid.shape
        npt = points.shape[1]
        points_weight = torch.ones([nc, npt]).to(feat_grid.device)

        # giving points outside the image zero weight
        points_weight[points[:, :, 0] < 0] = 0
        points_weight[points[:, :, 1] < 0] = 0
        points_weight[points[:, :, 0] > (w - 1)] = 0
        points_weight[points[:, :, 1] > (h - 1)] = 0

        points = points.unsqueeze(2).repeat([1, 1, 4, 1])
        # later used for calculating weight
        points_copy = points.detach().clone()

        # getting discrete grid location of pts in the camera image space
        points[:, :, 0, 0] = torch.floor(points[:, :, 0, 0])
        points[:, :, 0, 1] = torch.floor(points[:, :, 0, 1])
        points[:, :, 1, 0] = torch.floor(points[:, :, 1, 0])
        points[:, :, 1, 1] = torch.ceil(points[:, :, 1, 1])
        points[:, :, 2, 0] = torch.ceil(points[:, :, 2, 0])
        points[:, :, 2, 1] = torch.floor(points[:, :, 2, 1])
        points[:, :, 3, 0] = torch.ceil(points[:, :, 3, 0])
        points[:, :, 3, 1] = torch.ceil(points[:, :, 3, 1])
        grid_points = points.long()  # [nc, npt, 4, 2] (grid)

        # o─────────────o
        # │             │
        # │   x         │
        # │             │
        # │             │
        # │             │
        # │             │
        # │             │
        # o─────────────o
        # since we are taking modulo, points at the edge, i,e at h or w will be
        # mapped to 0. this will make their distance from the continous location
        # large and hence they won't matter. therefore we don't need an explicit
        # step to remove such points
        grid_points[:, :, :, 0] = torch.fmod(grid_points[:, :, :, 0], int(w))
        grid_points[:, :, :, 1] = torch.fmod(grid_points[:, :, :, 1], int(h))
        grid_points[grid_points < 0] = 0

        # getting normalized weight for each discrete location for pt
        # weight based on distance of point from the discrete location
        # [nc, npt, 4]
        points_dist = 1 / (torch.sqrt(torch.sum((points_copy - grid_points) ** 2, dim=-1)) + 1e-10)
        points_weight = points_weight.unsqueeze(-1) * points_dist
        _points_weight = torch.sum(points_weight, dim=-1, keepdim=True)
        _points_weight[_points_weight == 0.0] = 1
        # cached points_wei in select_feat_from_hm_cache
        points_weight = points_weight / _points_weight  # [nc, npt, 4]

        grid_points = grid_points.view(nc, 4 * npt, 2)  # [nc, 4 * npt, 2]
        # cached points in select_feat_from_hm_cache
        if self.token_format == 'xy':
            grid_points = (grid_points[:, :, 1] * w) + grid_points[:, :, 0]  # [nc, 4 * npt]
        elif self.token_format == 'hw':
            grid_points = (grid_points[:, :, 0] * w) + grid_points[:, :, 1]  # [nc, 4 * npt]
        else:
            raise ValueError("token_format should be 'xy' or 'hw'")

        # transforming indices from 2D to 1D to use pytorch gather
        feat_grid = feat_grid.permute(0, 2, 3, 1).view(nc, h * w, nw)  # [nc, h * w, nw]
        # [nc, 4 * npt, nw]
        points_val = self.batched_index_select(feat_grid, dim=1, index=grid_points)
        # tranforming back each discrete location of point
        points_val = points_val.view(nc, -1, 4, nw)
        # summing weighted contribution of each discrete location of a point
        points_val = torch.sum(points_val * points_weight.unsqueeze(-1), dim=2) # [nc, npt, nw]
        return points_val


class ChunkEmbedding(nn.Module):
    def __init__(self, n_embd: int, max_chunk_size: int, tokens: List[TokenType]):
        super().__init__()
        self.chunk_embed = nn.Embedding(max_chunk_size, n_embd)
        self.token_type_embed = nn.Embedding(len(tokens), n_embd)

    def forward(self, chk_ids, tk_ids):
        """
        > note the chunk embedding is shared across all tokens, more like a relative position embedding within each
        > set of chunks
        > for example, if chk_ids = [0, 0, 0, 1, 2, 2], then we have 3 sets of chunks, and we want to transform it into
        # [0,1,2, 0, 0,1], and then apply embedding layer

        chk_ids: (B | 1, L), chunk ids 
        tk_ids: (B, L), token ids
        return (B, L, embs)
        """
        chk_ids = chk_ids.long()
        tk_id_emb = self.token_type_embed(tk_ids.long())
        reg_indices = chk_ids.clone()
        for i in range(chk_ids.size(0)):
            reg_indices[i] = self.chk_ids_to_indices(chk_ids[i])
        if reg_indices.size(0) == 1:
            reg_indices = reg_indices.repeat(tk_ids.size(0), 1) 
        reg_emb = self.chunk_embed(reg_indices)
        return reg_emb + tk_id_emb
        
    @staticmethod
    def chk_ids_to_indices(ids):
        """
        ids: (L) 
        return: (L)
        this function looks obscure, but what it does is very simple, like the example above 
        input: [0,0,0, 1, 2,2], output: [0,1,2, 0, 0,1]
        input: [1,1,1, 2,2, 3,3,3, 4, 5,5,5,5,5], output: [0,1,2, 0,1, 0,1,2, 0, 0,1,2,3,4]
        (transform chunk ids to relative indices within each set)
        """
        dev, min_id = ids.device, ids.min()
        counts = torch.unique(ids, return_counts=True, sorted=True)[1]
        starts = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), counts[:-1].cumsum(0)])
        index = torch.bucketize(ids, torch.arange(min_id, min_id + len(counts), device=dev))
        return torch.arange(0, len(ids), device=dev) - starts[index]


#endregion Embedding ###############


#region Predictor


class TokenPredictorInterface(nn.Module):
    IS_CONTINUOUS = False
    def __init__(self, n_embd: int, token: TokenType, **kwargs):
        super().__init__()
        self.n_embd = n_embd
        self.token = token
    
    def sample(self, predicts_of_curr_regs: List[Union[Tensor, D.Distribution]], do_sample:bool, **extra_contexts) -> Tensor:
        raise NotImplementedError

    def forward(self, embs, log_prob=False, **extra_contexts) -> Union[Dict[str, List[Tensor]], D.Distribution]:
        pass

@register_token_predictor('gmm')
class GMMPredictor(TokenPredictorInterface):
    IS_CONTINUOUS = True

    def __init__(self, n_embd: int, token: TokenType, num_latents=1, low_var_eval=True, label_name='label', **kwargs):
        super().__init__(n_embd, token, **kwargs)
        self.num_latents = num_latents
        self.low_var_eval = low_var_eval
        self.label_name = label_name
        if num_latents == 1:
            out_features = 2 * token['dim']
            self.gauss_nll_loss = nn.GaussianNLLLoss()
        else:
            out_features = num_latents * 2 * token['dim'] + num_latents # means, scales, logits
        self.mlp = Mlp(in_features=n_embd, hidden_features=n_embd, out_features=out_features) 
    
    def forward(self, embs, log_prob=False, split_distributions=False, **extra_contexts) -> Union[Dict[str, List[Tensor]], D.Distribution]:
        """ embs: (*, d) 
        """
        base_shape = embs.shape[:-1]
        loss_dict = defaultdict(list)
        if self.num_latents > 1:
            out = self.mlp(embs)
            logits = out[..., :self.num_latents]
            means, raw_scales = out[..., self.num_latents:].chunk(2, dim=-1)
            scales = torch.exp(0.5 * raw_scales)
            if not self.training and self.low_var_eval:
                scales[:] = (1e-5 * means).abs()
            means = means.reshape(*base_shape, self.num_latents, self.token['dim'])
            scales = scales.reshape(*base_shape, self.num_latents, self.token['dim'])
            component_distribution = D.Normal(loc=means, scale=scales)
            component_distribution = D.Independent(component_distribution, 1)
            mixture_distribution = D.Categorical(logits=logits)
            dist = D.MixtureSameFamily(
                mixture_distribution=mixture_distribution,
                component_distribution=component_distribution,
            )
            if self.training:
                label = extra_contexts[self.label_name]
                loss_dict['nll_loss'].append(- dist.log_prob(label).mean())
                if log_prob:
                    loss_dict['log_prob'].append(dist.log_prob(label))
                return loss_dict
            else:
                if split_distributions:
                    dists = []
                    for i in range(base_shape[-1]): # seq len
                        component_distribution = D.Normal(loc=means[:, i], scale=scales[:, i])
                        component_distribution = D.Independent(component_distribution, 1)
                        mixture_distribution = D.Categorical(logits=logits[:, i])
                        dists.append(D.MixtureSameFamily(
                            mixture_distribution=mixture_distribution,
                            component_distribution=component_distribution,
                        ))
                    return dists
                else:
                    return dist
        else:
            means, raw_scales = self.mlp(embs).chunk(2, dim=-1)
            scales = torch.exp(0.5 * raw_scales)
            dist = D.Normal(means, scales)
            if self.training:
                label = extra_contexts['label']
                loss_dict['nll_loss'].append(self.gauss_nll_loss(means, label, scales ** 2))
                if log_prob:
                    loss_dict['log_prob'].append(dist.log_prob(label))
                return loss_dict
            else:
                if split_distributions:
                    return [D.Normal(means[:, i], scales[:, i]) for i in range(means.shape[1])]
                else:
                    return dist
    
    def sample(self, predicts_of_curr_regs: Union[Tensor, D.Distribution], do_sample:bool, **extra_contexts) -> Tensor:
        result = []
        for d in predicts_of_curr_regs:
            assert isinstance(d, D.Distribution)
            if do_sample:
                if self.low_var_eval and self.num_latents == 1:
                    samples = d.mean
                else:
                    samples = d.sample()
                result.append(samples)
            else:
                result.append(d.mean)
        return result


@register_token_predictor('class')
class ClassPredictor(TokenPredictorInterface):
    IS_CONTINUOUS = False

    def __init__(self, n_embd: int, token: TokenType, label_name='label', **kwargs):
        super().__init__(n_embd, token, **kwargs)
        self.linear = nn.Linear(n_embd, sum(token['dict_sizes']))
        self.ce_loss = nn.CrossEntropyLoss()
        self.ce_loss_wo_mean = nn.CrossEntropyLoss(reduction='none')
        self.label_name = label_name
    
    def forward(self, embs, log_prob=False, **extra_contexts) -> Union[Dict[str, List[Tensor]], D.Distribution]:
        """ embs: (*, d)
        """
        logits = self.linear(embs) # (bs, dict_sizes)
        loss_dict = defaultdict(list)
        max_dict_size = max(self.token['dict_sizes'])
        start, outputs = 0, []
        for j, size in enumerate(self.token['dict_sizes']):
            logits_j = logits[..., start:start+size]
            start += size
            if self.training: 
                label = extra_contexts[self.label_name]
                ce_loss = self.ce_loss(logits_j, label[..., j].long())
                loss_dict['ce_loss'].append(ce_loss)    
                if log_prob:
                    loss_dict['log_prob'].append(self.ce_loss_wo_mean(logits_j, label[..., j].long()))
            else:
                outputs.append(pad_last_dim(logits_j, max_dict_size, val=float('-inf')).unsqueeze(-2))
        if self.training:
            return loss_dict
        else:
            return torch.cat(outputs, dim=-2)
    
    def sample(self, predicts_of_curr_regs: List[Union[Tensor, D.Distribution]], do_sample:bool, **extra_contexts) -> Tensor:
        result = []
        for d in predicts_of_curr_regs:
            assert isinstance(d, Tensor)
            probs = F.softmax(d, dim=-1)
            if do_sample:
                samples = torch.multinomial(probs.flatten(0, -2), num_samples=1).reshape(*probs.shape[:-1])
            else:
                _, samples = torch.topk(probs, k=1, dim=-1)
                samples = samples.squeeze(-1)
            result.append(samples)
        return result


class ConvUpsample(nn.Module):
    """ from RVT2 and RAFT """
    def __init__(self, in_dim, out_dim, up_ratio, up_kernel=3, mask_scale=0.1, hidden_dim_mult=2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.up_ratio = up_ratio
        self.up_kernel = up_kernel
        self.mask_scale = mask_scale
        assert (self.up_kernel % 2) == 1
        hidden_dim = int(hidden_dim_mult * in_dim)
        self.net_out = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_dim, 3, padding=1),
        )
        mask_dim = (self.up_ratio**2) * (self.up_kernel**2) # (14 * 14) * 9
        self.net_mask = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, mask_dim, 1, padding=0),
        )

    def forward(self, x):
        """
        x: (bs, in_dim, h, w)
        return: (bs, out_dim, h*up_ratio, w*up_ratio)
        """
        bs, c, h, w = x.shape
        assert c == self.in_dim, c
        out_low = self.net_out(x)
        mask = self.mask_scale * self.net_mask(x)
        mask = mask.view(bs, 1, self.up_kernel**2, self.up_ratio, self.up_ratio, h, w)
        mask = torch.softmax(mask, dim=2) # bs, 1, 9, 14, 14, h, w
        out = F.unfold(
            out_low,
            kernel_size=[self.up_kernel, self.up_kernel],
            padding=self.up_kernel // 2,
        )
        out = out.view(bs, self.out_dim, self.up_kernel**2, 1, 1, h, w)
        out = torch.sum(out * mask, dim=2)
        out = out.permute(0, 1, 4, 2, 5, 3)
        out = out.reshape(bs, self.out_dim, h * self.up_ratio, w * self.up_ratio)
        return out


@register_token_predictor('upsample_from_2d_attn')
class Upsample2DAttnPredictor(TokenPredictorInterface):
    IS_CONTINUOUS = False

    def __init__(self, n_embd: int, token: TokenType, attn_with: Union[str, Tuple[int, ...]], upscale_ratio: int, token_format='xy', label_name='label', 
                corr_dim=-1, hidden_dim_mult=2, **kwargs):
        super().__init__(n_embd, token, **kwargs)
        if corr_dim < 0: 
            corr_dim = n_embd
        else:
            self.corr_proj = nn.Sequential(
                        nn.Conv2d(n_embd, corr_dim, 1), 
                        nn.BatchNorm2d(corr_dim),
                        nn.ELU(),                               
                        nn.Conv2d(corr_dim,  corr_dim, 5, padding=2, groups=corr_dim),
                        nn.BatchNorm2d(corr_dim),
                        nn.ELU(),    
                        nn.Conv2d(corr_dim, corr_dim, 1),
                        nn.BatchNorm2d(corr_dim),
                        nn.ELU())
        self.upsample = ConvUpsample(corr_dim, 1, upscale_ratio, hidden_dim_mult=hidden_dim_mult)
        self.corr_dim = corr_dim
        if isinstance(attn_with, str):
            self.attn_with = attn_with 
        else:
            assert len(attn_with) == 3
            self.register_parameter('attn_with', nn.Parameter(0.02 * torch.randn(1, *attn_with)))
        self.token_format = token_format
        self.label_name = label_name
        self._cross_entropy_loss = nn.CrossEntropyLoss()
        self._cross_entropy_loss_wo_mean = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, embs, log_prob=False, **extra_contexts):
        """ embs: (*, d) 
        """
        if isinstance(self.attn_with, str):
            feats = extra_contexts[self.attn_with] # (B, self.n_embd, H, W)
        else:
            feats = self.attn_with # (1, self.n_embd, H, W)
        embs = embs.reshape(-1, self.n_embd, 1, 1)
        corr = embs * feats
        if hasattr(self, 'corr_proj'): corr = self.corr_proj(corr)
        spatial_logits_map = self.upsample(corr) # (B, 1, H, W)
        if self.training:
            label = extra_contexts[self.label_name]
            if label.numel() == spatial_logits_map.numel():
                label = label.flatten(1)
            else:
                _, _, h, w = spatial_logits_map.shape
                if self.token_format == 'xy':
                    label = (label[..., 1] * w) + label[..., 0]
                elif self.token_format == 'hw':
                    label = (label[..., 0] * w) + label[..., 1]
                else:
                    raise ValueError("token_format should be 'xy' or 'hw'")
            
            result = {'2d_ce_loss': [self._cross_entropy_loss(spatial_logits_map.flatten(1), label)]}
            if log_prob:
                result['log_prob'] = [self._cross_entropy_loss_wo_mean(spatial_logits_map.flatten(1), label)]
            return result
        else:
            return spatial_logits_map
    
    def sample(self, predicts_of_curr_regs: List[Union[Tensor, D.Distribution]], do_sample:bool, **extra_contexts) -> Tensor:
        result = []
        for d in predicts_of_curr_regs:
            assert isinstance(d, Tensor)
            _, h, w = d.shape
            d = d.flatten(1)
            probs = F.softmax(d, dim=-1)
            if do_sample:
                samples = torch.multinomial(probs, num_samples=1)
            else:
                _, samples = torch.topk(probs, k=1, dim=-1)
            pred_h, pred_w = samples.flatten() // w, samples.flatten() % w
            if self.token_format == 'xy':
                samples = torch.cat([pred_w[:, None], pred_h[:, None]], dim=-1)
            elif self.token_format == 'hw':
                samples = torch.cat([pred_h[:, None], pred_w[:, None]], dim=-1)
            else: 
                raise ValueError("token_format should be 'xy' or 'hw'")
            result.append(samples)
        return result

#endregion Predictor ###############


#region Main Model

class AutoRegressivePolicy(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        for i, tk in enumerate(cfg.tokens): assert tk['id'] == i
        
        self.token_coder = TokenCoder(cfg.tokens)
        self.chunk_embedder = ChunkEmbedding(cfg.n_embd, cfg.max_chunk_size, cfg.tokens)

        self.token_embedders = nn.ModuleList()
        self.token_name_2_ids = {}
        self.f_token_name_2_ids = lambda name: self.token_name_2_ids.get(name, name)
        for tk_id, tk in enumerate(cfg.tokens):
            assert tk['embedding'] in register_token_embedding.map, f"token embedding type: {tk['embedding']} not found!"
            self.token_name_2_ids[tk['name']] = tk_id
            self.token_embedders.append(register_token_embedding.map[tk['embedding']](cfg.n_embd, tk, **tk['embedding_kwargs']))

        self.token_predictors: List[TokenPredictorInterface] = nn.ModuleList()
        for tk in cfg.tokens:
            assert tk["predictor"] in register_token_predictor.map, f"token predictor type: {tk['predictor']} not found!"
            self.token_predictors.append(nn.Identity() if tk['is_control'] else register_token_predictor.map[tk['predictor']](cfg.n_embd, tk, **tk['predictor_kwargs']))

        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.n_embd)

        self.blocks = nn.ModuleList()
        for layer_cfg in cfg.layers: 
            layer = ChunkTransformerLayer(
                cfg.n_embd, layer_cfg['n_head'], mlp_ratio=layer_cfg['mlp_ratio'], mlp_dropout=layer_cfg['mlp_dropout'],
                attn_kwargs=layer_cfg['attn_kwargs'], cond_attn_kwargs=layer_cfg['cond_attn_kwargs'],
                conditional=layer_cfg['condition_on'], AdaLN=layer_cfg.get('AdaLN', False), norm_before_AdaLN=layer_cfg.get('norm_before_AdaLN', False)
            )
            self.blocks.append(layer)

        self.drop = nn.Dropout(cfg.embd_pdrop)
        if cfg.layer_norm_every_block:
            self.layer_norms = nn.ModuleList([nn.LayerNorm(cfg.n_embd) for _ in range(len(cfg.layers))])
        else:
            self.final_ln = nn.LayerNorm(cfg.n_embd)

        self.cfg = cfg
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
                    torch.nn.init.ones_(module.weight)           
        self.apply(_basic_init)

        for block in self.blocks:
            if block.AdaLN:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def token_codes_to_embeddings(self, tk_codes: Tensor, tk_ids: Tensor, **extra_contexts):
        dev = tk_codes.device
        shape = tk_ids.shape
        embs = torch.zeros(list(shape) + [self.cfg.n_embd, ], dtype=torch.float, device=dev)
        for i in map(int, torch.unique(tk_ids)):
            mask = tk_ids == i
            embedder = self.token_embedders[i]
            embs[mask, :] = embedder(tk_codes[mask], **extra_contexts).to(embs.dtype)
        return embs
    
    @staticmethod
    def filter_context(curr_chk_id, contexts):
        """
        curr_chk_id: int
        contexts: str -> (tensors | (int -> tensor) | (frozenset of int -> tensor))
        return: dict of str to tensor
        """
        chk_contexts = {}
        for ki, vi in contexts.items():
            if isinstance(vi, Tensor) or vi is None: chk_contexts[ki] = vi
            else:
                for kj, vj in vi.items():
                    if str(curr_chk_id) in str(kj):
                        chk_contexts[ki] = vj
                        break
                if ki not in chk_contexts and 'default' in vi: chk_contexts[ki] = vi['default']
        return chk_contexts

    def forward(self, embs: Union[Tensor, Tuple[Tensor, Tensor]], chk_ids: Tensor, 
                layer_ids: Optional[List[int]]=None, contexts: Dict[str, Tensor]={}, 
                dependency_attn_mask: Optional[Tensor]=None, 
                training: bool=None):
        if layer_ids is None: layer_ids = list(range(len(self.blocks)))
        if training is None: training = self.training
        if training: dev, (bs, L) = embs[0].device, embs[0].shape[:2]
        else: dev, (bs, L) = embs.device, embs.shape[:2]

        pos_emb = self.pos_emb(torch.arange(0, L, dtype=torch.long, device=dev))[None, ...]
        if training:
            train_masks = ChunkTransformerLayer.train_attn_masks(chk_ids)
            embs = [self.drop(e + pos_emb) for e in embs]
            for layer_id in layer_ids:
                block: ChunkTransformerLayer = self.blocks[layer_id]
                cond = contexts[block.conditional] if block.conditional else None
                embs = block.forward_train(embs, cond, train_masks, dependency_attn_mask=dependency_attn_mask)
                if self.cfg.layer_norm_every_block:
                    embs = [self.layer_norms[layer_id](e) for e in embs] 
            if not self.cfg.layer_norm_every_block:
                embs = [self.final_ln(e) for e in embs]
        else:
            eval_mask = ChunkTransformerLayer.eval_attn_mask(chk_ids)
            if dependency_attn_mask is not None: 
                eval_mask = eval_mask & dependency_attn_mask
            embs = self.drop(embs + pos_emb)
            for layer_id in layer_ids:
                block: ChunkTransformerLayer = self.blocks[layer_id]
                cond = contexts[block.conditional] if block.conditional else None
                embs = block.forward_inference(embs, cond, eval_mask)
                if self.cfg.layer_norm_every_block:
                    embs = self.layer_norms[layer_id](embs)        
            if not self.cfg.layer_norm_every_block:
                embs = self.final_ln(embs)
        return embs

    def compute_loss(self, tks: Tensor, chk_ids: Optional[Tensor]=None, valid_tk_mask: Tensor=None, 
                     skip_tokens: List[int] = [], 
                     block_attn_directions: AttnDirectionsType=[],
                     match_layer: str = "",
                     contexts: Dict[str, Tensor]={}, log_prob=False):
        """
        tks: (bs, L, d+1), where d = max(dims of all actions), the last dimension is tk_ids
        tk_ids: (bs, L)
        chk_ids: (bs | 1, L)
        skip_tokens: list of token ids to skip in computing loss
        valid_tk_mask: (bs, L), False means not computing loss on that token
        block_attn_directions: list of (str, str) | (int, int), mask the attention from token i to token j
        match_layer: str 
        contexts: dict of tensors, used for embedding, condition, and prediction 
        """
        tk_ids, tks = tks[:, :, -1], tks[:, :, :-1]
        dev, batch_size = tks.device, len(tks)
        losses = defaultdict(list)
        if chk_ids is None: chk_ids = torch.arange(0, tk_ids.size(1), device=dev)[None, ...]
        if len(chk_ids.shape) == 1: chk_ids = chk_ids[None, ...]
        assert chk_ids.size(1) == tk_ids.size(1), "chunk ids and token should have the same length"
        tk_codes = self.token_coder.encode(tks, tk_ids)

        dependency_attn_mask = ChunkTransformerLayer.dependency_attn_mask(tk_ids, 
                    map2(self.f_token_name_2_ids, block_attn_directions)) if block_attn_directions else None

        embs_star = self.token_codes_to_embeddings(tk_codes, tk_ids, **contexts) 
        embs_hat = self.chunk_embedder(chk_ids, tk_ids)

        if match_layer:
            layer_ids = [i for i, ln in enumerate(self.cfg.layers) if match_layer in ln['name']]
        else:
            layer_ids = list(range(len(self.blocks)))

        # interleave forward
        embs_star, embs_hat = self([embs_star, embs_hat], chk_ids, contexts=contexts, layer_ids=layer_ids,
                                   dependency_attn_mask=dependency_attn_mask, training=True)

        cond_log_probs = torch.zeros(batch_size, tks.shape[1], device=dev) if log_prob else None
        for i in map(int, tk_ids.unique()):
            token = self.cfg.tokens[i]
            if token.get('is_control', False): continue
            if i in skip_tokens: continue
            mask = tk_ids == i
            if valid_tk_mask is not None:
                mask = mask & valid_tk_mask
            
            _tks = tk_codes if token['is_continuous'] and not self.token_predictors[i].IS_CONTINUOUS else tks
            is_training = self.token_predictors[i].training
            self.token_predictors[i].train(True)
            loss_dict = self.token_predictors[i](embs_hat[mask], label=_tks[mask][..., :token['dim']], **contexts)
            self.token_predictors[i].train(is_training)
            if log_prob:
                ll = sum(loss_dict.pop('log_prob'))
                cond_log_probs[mask] = ll
            for k, v in loss_dict.items(): 
                losses[f'{token["name"]}.{k}'] += v

        loss_dict = {k: sum(v) / len(v) for k, v in losses.items()}
        if log_prob:
            return loss_dict, cond_log_probs
        else:
            return loss_dict
    
    @torch.no_grad()
    def generate(self, prompt_tks: Tensor, future_tk_chk_ids: List[IncompleteToken], 
                sample: bool=False, contexts: Dict[str, PerChunk[Tensor]]={},
                block_attn_directions: AttnDirectionsType=[],
                match_layer: PerChunk[str]= "",
                sample_function: PerChunk[SampleFunctionT]={}):    
        """
        prompt_tks: (bs, L, d+1), where d = max(dims of all actions), the last dimension is tk_ids
        future_tk_chk_ids: list of dict(tk_id, chk_id, tk_val), tk_val is optional and only shall be given for control token 
        sample: bool
        contexts: dict of tensors, used for embedding, condition, prediction. customizable based on the reg id.
        block_attn_directions: list of (str, str) | (int, int), mask the attention from token i to token j
        match_layer: str, customizable based on the reg id, used for filter layers
        sample_function: dict of chk_id  or set of chk_ids to sample_function. Used to customize the sampling process for each set of chunks.

        return: completed sequence (bs, L+len(future_tk_chk_ids), d+1), where d = max(dims of all actions), the last dimension is tk_ids
        """
        assert not self.training, "model should be in eval mode during generation"
        future_tk_chk_ids = deepcopy(future_tk_chk_ids)
        dev, batch_size = prompt_tks.device, len(prompt_tks)
        tk_ids, tks = prompt_tks[:, :, -1], prompt_tks[:, :, :-1]

        sample_function =  flatten_per_chunk_dict(sample_function) if isinstance(sample_function, dict) else sample_function
        match_layer = match_layer if isinstance(match_layer, str) else flatten_per_chunk_dict(match_layer)

        chk_ids = torch.arange(0, prompt_tks.size(1), device=dev)[None, ...]
        tk_codes = self.token_coder.encode(tks, tk_ids)

        def to_seq(codes, ids):
            vals = self.token_coder.decode(codes, ids)
            return torch.cat([vals, ids[..., None]], dim=-1)
        
        # running tensors: tk_codes, chk_ids, tk_ids
        while len(future_tk_chk_ids) > 0:
            curr_chk_id = future_tk_chk_ids[0]['chk_id']
            assert curr_chk_id >= prompt_tks.size(1), "future chunk id should >= the prompt length"
            curr_tokens: List[TokenType] = []
            next_chunk: List[IncompleteToken] = []
            while len(future_tk_chk_ids) > 0 and future_tk_chk_ids[0]['chk_id']== curr_chk_id:
                next_chunk.append(future_tk_chk_ids.pop(0))
                curr_tokens.append(self.cfg.tokens[next_chunk[-1]['tk_id']])
            
            next_tk_codes = torch.zeros(batch_size, len(next_chunk), tk_codes.size(-1), device=dev, dtype=tk_codes.dtype)

            next_chk_ids = torch.as_tensor([v['chk_id'] for v in next_chunk], device=dev)[None, :]
            next_tk_ids_lst = [v['tk_id'] for v in next_chunk]
            next_tk_ids = torch.as_tensor(next_tk_ids_lst, device=dev, dtype=tk_ids.dtype)[None, :].repeat(batch_size, 1)
            chk_ids = torch.cat([chk_ids, next_chk_ids], dim=1)

            if all([curr_token.get('is_control', False) for curr_token in curr_tokens]): 
                next_tk_codes[:, :, :1] = torch.as_tensor([v['tk_val'] for v in next_chunk], device=dev)[None, :, None]
                tk_codes = torch.cat([tk_codes, next_tk_codes], dim=1)
                tk_ids = torch.cat([tk_ids, next_tk_ids], dim=1)
                continue

            chk_contexts = self.filter_context(curr_chk_id, contexts)

            prompt_embs = self.token_codes_to_embeddings(tk_codes, tk_ids, **chk_contexts)
            chunk_embs = self.chunk_embedder(next_chk_ids, next_tk_ids)
            embs = torch.cat([prompt_embs, chunk_embs], dim=1) 

            match_layer_ = match_layer if isinstance(match_layer, str) else match_layer.get(curr_chk_id, "")
            if match_layer_:
                layer_ids = [i for i, ln in enumerate(self.cfg.layers) if match_layer in ln['name']]
            else:
                layer_ids = list(range(len(self.blocks)))

            dependency_attn_mask = ChunkTransformerLayer.dependency_attn_mask(torch.cat([tk_ids, next_tk_ids], dim=1), 
                                                                              map2(self.f_token_name_2_ids, block_attn_directions)) if block_attn_directions else None
            embs = self(embs, chk_ids, contexts=chk_contexts, layer_ids=layer_ids,
                        dependency_attn_mask=dependency_attn_mask, training=False)
            embs = embs[:, -len(next_chunk):, :] # the current chunk set

            predicts = [None] * len(next_tk_ids_lst)
            for tk_id_key, _ in itertools.groupby(next_tk_ids_lst):
                _indices = [_i for _i, v in enumerate(next_tk_ids_lst) if v == tk_id_key]
                predict_output = self.token_predictors[tk_id_key](embs[:, _indices, :], split_distributions=True, **chk_contexts)
                if isinstance(predict_output, Tensor):
                    predict_output = predict_output.reshape(batch_size, len(_indices), *predict_output.shape[2:])
                    for _i, _ind in enumerate(_indices): predicts[_ind] = predict_output[:, _i]
                else:
                    for _i, _ind in enumerate(_indices): predicts[_ind] = predict_output[_i]
            
            if callable(sample_function):
                next_tk_codes = sample_function(predicts)
            elif curr_chk_id in sample_function:
                next_tk_codes = sample_function[curr_chk_id](predicts)
            else:
                next_tk_codes, start = [], 0
                for tk_id_key, group in itertools.groupby(next_tk_ids_lst):
                    count = len(list(group))
                    output = self.token_predictors[tk_id_key].sample(predicts[start:start+count], do_sample=sample, **chk_contexts)
                    output_val_or_code = torch.cat([o[:, None, :] for o in output], dim=1)
                    if self.token_coder.need_encoding(self.cfg.tokens[tk_id_key], self.token_predictors[tk_id_key].IS_CONTINUOUS):
                        output_code = self.token_coder.encode_ith(output_val_or_code, tk_id_key)
                    else:
                        output_code = output_val_or_code
                    next_tk_codes.append(output_code)
                    start += count

                next_tk_codes = cat_uneven_blc_tensors(*next_tk_codes)
            tk_codes = cat_uneven_blc_tensors(tk_codes, next_tk_codes)
            tk_ids = torch.cat([tk_ids, next_tk_ids], dim=1)
        return to_seq(tk_codes, tk_ids)

#endregion Main Model ###############


if __name__ == "__main__":
    import cv2
    import numpy as np
    import torch
    from torch import nn
    import os
    from tqdm.auto import tqdm
    from torchvision.utils import make_grid
    from torchvision.transforms.functional import to_pil_image
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from torchvision import datasets, transforms

    class MnistDataset(Dataset):
        def __init__(self, mnist, len, img_size=28, threshold=0.5):
            self.mnist = mnist
            self.len = len
            self.img_size = img_size
            self.threshold = threshold

        def __len__(self):
            return len(self.mnist)

        def __getitem__(self, idx):
            img, label = self.mnist[idx]
            img = img.reshape(28, 28)
            if self.img_size != 28:
                img = cv2.resize(img.numpy(), (self.img_size, self.img_size))
                img = torch.from_numpy(img)
            indices = (img > self.threshold).nonzero()
            seq = torch.zeros(self.len, 2, dtype=torch.long)
            if len(indices) >= self.len-1:
                seq[1:] = indices[:self.len-1]
            else:
                seq[1:1+len(indices)] = indices
                seq[1+len(indices):] = indices[[-1]]
            return seq, label

        def from_seq(self, seqs):
            imgs = []
            for seq in seqs:
                img = torch.zeros([self.img_size, self.img_size], dtype=torch.uint8)
                img[seq[:, 0], seq[:, 1]] = 255
                imgs.append(img[None, ...])

            fig = to_pil_image(make_grid(imgs))
            return fig

    class config:
        class data:
            length: int = 140

        class train:
            device: str =  "cuda:0"
            num_workers: int = 0
            max_iters: int = 5000
            batch_size: int = 64
            learning_rate: float = 5e-4
            betas: Tuple[float, float] = (0.9, 0.95)
            weight_decay: float = 0.1 # only applied on matmul weights
            grad_norm_clip: float = 1.0
    
    
    iter_num = 0
    transform=transforms.Compose([transforms.ToTensor()])
    mnist = datasets.MNIST('./mnist', train=True, download=True, transform=transform)
    mnist_drawing_dataset = MnistDataset(mnist, config.data.length)

    model = AutoRegressivePolicy(ModelConfig(
        n_embd = 64,
        embd_pdrop = 0.1,
        max_seq_len = config.data.length * 2,
        max_chunk_size= 1,
        tokens = [
                TokenType.make(id=0, name='control', is_control=True),
                TokenType.make(id=1, name='h', dict_sizes=[28]), 
                TokenType.make(id=2, name='w', dict_sizes=[28]), 
        ],
        layers = [
            dict(n_head = 4,
                num_layers = 4,
                attn_kwargs = dict(attn_pdrop=0.1, resid_pdrop=0.1),
                cond_attn_kwargs = dict(attn_pdrop=0.1, resid_pdrop=0.1),
                mlp_dropout = 0.1,
                AdaLN=True, condition_on="class_emb",
                mlp_ratio = 4.0)
        ] * 4
    ))
    model.condition_embedding = nn.Embedding(10, model.cfg.n_embd)
    model = model.to(config.train.device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate, betas=config.train.betas)

    
    train_loader = DataLoader(
        mnist_drawing_dataset,
        shuffle=True,
        pin_memory=True,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
    )

    data_iter = iter(train_loader)
    with tqdm(total=config.train.max_iters, desc="Training a Binary MINIST Generator with ARP") as pbar:
        while True:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(config.train.device) for t in batch]
            x, y = batch
            y_emb = model.condition_embedding(y)
            
            bs = len(x)
            tk_vals = x.reshape(bs, -1, 1)[:, 1:, :] # [ctrl][x][y][x][y]...
            tk_ids = torch.as_tensor([0] + [1, 2] * (config.data.length - 1)).to(x.device)
            tk_ids = tk_ids.reshape(1, -1, 1).repeat(bs, 1, 1)
            tks = torch.cat([tk_vals, tk_ids], dim=-1)

            chk_ids = torch.arange(0, tks.size(1), device=tks.device)[None, :] # generate one at a time
            loss_dict = model.compute_loss(tks, chk_ids, contexts={'class_emb': y_emb[:, None, :]})
            loss = sum(loss_dict.values()) 

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_norm_clip)
            optimizer.step()

            if iter_num % 100 == 0:
                loss_dict = {k: round(v.item(), 5) for k,v in loss_dict.items()}
                tqdm.write(f"iter {iter_num}: {loss_dict}")
            iter_num += 1
            pbar.update()

            # termination conditions
            if iter_num >= config.train.max_iters:
                break
    
    model = model.eval()
    device = config.train.device
    y = torch.arange(0, 10).to(device)
    y_emb = model.condition_embedding(y)
    y_emb = y_emb[:, None, :]

    prompt_tk_vals = torch.full([10, 1, 1], fill_value=0.0, device=device).float()
    prompt_tk_ids = torch.as_tensor([model.token_name_2_ids['control'], ] * 10, device=device)
    prompt_tks = torch.cat([prompt_tk_vals, prompt_tk_ids.reshape(10, 1, 1)], dim=-1)

    future_tk_ids = [model.token_name_2_ids['h'], model.token_name_2_ids['w']] * (config.data.length - 1)
    future_chk_ids = list(range(prompt_tks.size(1), prompt_tks.size(1) + len(future_tk_ids)))
    future_tk_reg_ids = [{'tk_id':tk_id, 'chk_id': chk_id} for tk_id, chk_id in zip(future_tk_ids, future_chk_ids)]

    for img_id in tqdm(range(10), "Generating images to `mnist_generated_arp`"):
        pred_tk_vals = model.generate(prompt_tks, future_tk_reg_ids, contexts={ 'class_emb': y_emb}, sample=True)
        pred_tk_vals = pred_tk_vals[:, 1:, 0].cpu().long().reshape(10, -1, 2)

        imgs = []
        for i in range(10):
            img = torch.zeros([28, 28], dtype=torch.uint8)
            img[pred_tk_vals[i, :, 0], pred_tk_vals[i, :, 1]] = 255
            imgs.append(img[None, ...])

        fig = to_pil_image(make_grid(imgs))
        os.makedirs('mnist_generated_arp', exist_ok=True)
        fig.save(f'mnist_generated_arp/{img_id}.png')