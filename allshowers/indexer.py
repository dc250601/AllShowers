import torch
from torch import nn

import torch
from torch import nn

from .triton_kernels import triton_mask_maker, triton_custom_mask, triton_topk
import triton
import triton.language as tl

class Indexer(torch.nn.Module):
    def __init__(self,
                 hidden_dim,
                 head_dim,
                 n_heads,
                 num_layer_cond = 4,
                 TOP_K = 128,
                 topk_block_size = 1024,
                 USE_FLAT_MASK = False
                ):
        
        super().__init__()

        self.Wq = nn.Linear(hidden_dim,head_dim*n_heads)
        self.Wk = nn.Linear(hidden_dim,head_dim)
        self.Ww = nn.Linear(hidden_dim,n_heads)

        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.n_heads = n_heads

        self.num_layer_cond = num_layer_cond
        self.TOP_K = TOP_K

        self.topk_block_size = topk_block_size
        self.USE_FLAT_MASK = USE_FLAT_MASK

    @torch.compile(dynamic=True)
    def get_score(self,
                  Q,
                  K,
                  W,
                  b_idx,
                  q_idx,
                  k_idx):
        
        query_idx = b_idx * Q.shape[1] + q_idx
        key_idx = b_idx * K.shape[1] + k_idx
    
        sub_Q = Q.flatten(0, 1).index_select(0, query_idx)
        sub_K = K.flatten(0, 1).index_select(0, key_idx)
        sub_W = W.flatten(0, 1).index_select(0, query_idx)
    
        s = (sub_Q * sub_K[:, None, :]).sum(dim=-1)
        s = torch.relu(s)
    
        return (s * sub_W).sum(dim=-1)
    

    
    def prepare_neighbour_mask(self,
                               mask,
                               layer
                              ):
        
        layer = layer[:,:,0]
        mask = mask[:,:,0]
        all_mask = torch.empty((mask.shape[0], mask.shape[1],mask.shape[1]),dtype=torch.bool,device=mask.device)
        grid = (all_mask.shape[0],all_mask.shape[1])
        
        t_kernel = triton_mask_maker[grid](
            layer_ptr = layer,
            mask_ptr = mask,
            output_ptr = all_mask,
            NUM_LAYER_COND = 4,
            SEQUENCE_LENGTH =  mask.shape[1],
            BATCH_SIZE =  mask.shape[0]
        )

        return all_mask
        

    def prepare_score(self,
                x,
                neighbour_mask,
                     ):

        Q = self.Wq(x)

        Q = Q.reshape(x.shape[0],
                      x.shape[1],
                      self.n_heads,
                      self.head_dim)
        
        K = self.Wk(x)
        
        W = self.Ww(x)

        b,q,v = torch.where(neighbour_mask)
        
        score = self.get_score(Q,K,W,b,q,v)

        return score,b,q,v


    def prepare_topk(self,scores,b,q,v):
        
        b_idx = get_begin_idx(b)
        q_idx = get_begin_idx(q)
        
        start_idx = []
        stop_idx = []

        start_idx = q_idx
        stop_idx = torch.concat((q_idx[1:],torch.tensor([len(q)],device =q_idx.device)))
        
        output = torch.full_like(v,-1)
        grid = (len(start_idx),)

        t_kernel = triton_topk[grid](
            v,
            scores,
            output,
            start_idx,
            stop_idx,
            TOP_K=self.TOP_K,
            BLOCK_SIZE=self.topk_block_size,
        )

        mask = output !=-1
        b_new,q_new,v_new = b[mask],q[mask],v[mask]
        index_vec = torch.concatenate([b_new[:,None],q_new[:,None],v_new[:,None]],-1)

        return index_vec

    
    def get_topk_mask(self,x,index_vec):
        all_mask = torch.empty((x.shape[0],x.shape[1],x.shape[1]),dtype=torch.bool,device=x.device)
        all_mask[...] = False

        BLOCK_SIZE = 8192
        grid = (triton.cdiv(len(index_vec), BLOCK_SIZE),)
        
        t_kernel = triton_custom_mask[grid](
            mask_stream_ptr= index_vec,
            output_ptr = all_mask,
            n_elements =  len(index_vec),
            BLOCK_SIZE =  BLOCK_SIZE, ## Number of streams to process
            SEQUENCE_LENGTH = x.shape[1],
            BATCH_SIZE =  x.shape[0])

        return all_mask
    
    def forward(
        self,
        x,
        mask,
        layer,
    ):
        
        neighbour_mask = self.prepare_neighbour_mask(mask,layer)
        score,b,q,v =  self.prepare_score(x,neighbour_mask)

        if self.training:
            return score,b,q,v
        else:
            if self.USE_FLAT_MASK:
                return self.prepare_topk(score,b,q,v)
            else:
                index_vec = self.prepare_topk(score,b,q,v)
                top_mask = self.get_topk_mask(x,index_vec)
                return top_mask,neighbour_mask
        

    


def get_begin_idx(t):
    change_indices = torch.where(t[1:] != t[:-1])[0] + 1
    begin_idx = torch.cat([torch.tensor([0],device = t.device),change_indices])
    return begin_idx