import torch
from torch import nn

class Indexer(torch.nn.Module):
    def __init__(self,
                 hidden_dim,
                 head_dim,
                 n_heads,
                 num_layer_cond = 4):
        
        super().__init__()

        self.Wq = nn.Linear(hidden_dim,head_dim*n_heads)
        self.Wk = nn.Linear(hidden_dim,head_dim)
        self.Ww = nn.Linear(hidden_dim,n_heads)

        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.n_heads = n_heads

        self.num_layer_cond = num_layer_cond


    @torch.compile(fullgraph=True)
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

    
    @torch.compile(fullgraph=True)
    def compute_neighbour_mask(self,
                               padding_mask,
                               layer):
    
        padding_mask = padding_mask.flatten(1)
        layer = layer.flatten(1)
        
        if self.num_layer_cond < 0:
    
            def mask_fn(b, h, q_idx, kv_idx):
                return padding_mask[b, q_idx] & padding_mask[b, kv_idx]
        else:
    
            def mask_fn(b, h, q_idx, kv_idx):
                lower_bound = (
                    layer[b, q_idx] - layer[b, kv_idx] >= -1 * (self.num_layer_cond + 1) // 2
                )
                upper_bound = layer[b, q_idx] - layer[b, kv_idx] <= self.num_layer_cond // 2
                not_padding = padding_mask[b, q_idx] & padding_mask[b, kv_idx]
                return lower_bound & upper_bound & not_padding
        
        return mask_fn


    def forward(self,
                x,
                padding_mask,
                layer):

        Q = self.Wq(x)

        Q = Q.reshape(x.shape[0],
                      x.shape[1],
                      self.n_heads,
                      self.head_dim)
        
        K = self.Wk(x)
        
        W = self.Ww(x)


        batch,query,key = torch.meshgrid(torch.arange(0,x.shape[0],device=x.device),
                                     torch.arange(x.shape[1],device=x.device),
                                     torch.arange(x.shape[1],device=x.device),
                                    )

        mask_func = self.compute_neighbour_mask(padding_mask=padding_mask,
                                                layer=layer
                                               )
        
        mask = mask_func(batch,0,query,key)
        
        b,q,v = torch.where(mask)
        score = self.get_score(Q,K,W,b,q,v)

        return score,b,q,v



def get_begin_idx(t):
    change_indices = torch.where(t[1:] != t[:-1])[0] + 1
    begin_idx = torch.cat([torch.tensor([0],device = t.device),change_indices])
    return begin_idx


