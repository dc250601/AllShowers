import torch
import warnings
import torch.nn as nn
from .triton_kernels import triton_sparse_attention
from .indexer import Indexer
import torch.nn.functional as F

def get_begin_idx(t):
    change_indices = torch.where(t[1:] != t[:-1])[0] + 1
    begin_idx = torch.cat([torch.tensor([0],device = t.device),change_indices])
    return begin_idx


class MultiHeadSparseAttention:
    """
    Flex attention but on steriods. All calculations are flattened.
    """
    def __init__(self,
                 batch_size = 64,
                 sequence_length = 6016,
                 embed_dim = 64,
                 n_heads = 2,
                 use_projection = False,
                 max_qk_pairs = 128,
                ):
        
        self.W_q = None
        self.W_k = None
        self.W_v = None
        self.W_o = None

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.use_projection = use_projection

        self.max_qk_pairs = max_qk_pairs
        self.head_dim = embed_dim // n_heads

    def InitializeWithRandomWeights(self):
        
        self.W_q = torch.randn(self.embed_dim, self.embed_dim) / embed_dim**0.5
        self.W_k = torch.randn(self.embed_dim, self.embed_dim) / embed_dim**0.5
        self.W_v = torch.randn(self.embed_dim, self.embed_dim) / embed_dim**0.5
        self.W_o = torch.randn(self.embed_dim, self.embed_dim) / embed_dim**0.5
    
    def __call__(self,
                 mask,
                 X = None,
                 Q = None,
                 K = None,
                 V = None
                ):
        
        if self.W_q is None and self.use_projection:
            warnings.warn("Projection weights not initialized and projection is set to true !!! Using random weights")
            self.InitializeWithRandomWeights()

        if self.use_projection != True:
            assert (Q is not None) & (K is not None) & (V is not None), "Q,K,V matrices cannot be None when projetion is false"
        else:
            assert(X is not None), "X matrix not provided"
            
            assert len(X.shape) == 3, f"Expected a 3D tensor recieved a tensor of size {X.shape}"
            assert X.shape[0] == self.batch_size, "Batch size set incorectly"
            assert X.shape[1] == self.sequence_length, "Sequence length set incorrectly"
            assert X.shape[2] == self.embed_dim, "Embedding size did not match"


        mask_b,mask_q,mask_kv = mask[:,0].contiguous(), mask[:,1].contiguous(), mask[:,2].contiguous()

        if self.use_projection:
            Q = X @ self.W_q
            K = X @ self.W_k
            V = X @ self.W_v
        
        Q = Q.reshape(self.batch_size, self.sequence_length, self.n_heads, self.head_dim)
        K = K.reshape(self.batch_size, self.sequence_length, self.n_heads, self.head_dim)
        V = V.reshape(self.batch_size, self.sequence_length, self.n_heads, self.head_dim)

        v_out = self.__sparse_attention_kernel(Q,
                                               K,
                                               V,
                                               mask_b,
                                               mask_q,
                                               mask_kv)

        return v_out

    def __sparse_attention_kernel(self,
                                  Q,
                                  K,
                                  V,
                                  mask_b,
                                  mask_q,
                                  mask_kv):



        query_begin_pos = get_begin_idx(mask_q)
        n_ops = len(mask_b)
        query_end_pos = torch.concatenate([query_begin_pos[1:],torch.tensor([n_ops], device = Q.device)])
        v_out_triton = torch.zeros_like(Q)
        
        grid = (query_begin_pos.shape[0],self.n_heads)
        
        t_kernel = triton_sparse_attention[grid](
            Query = Q,
            Key = K,
            Value = V,
        
            Output = v_out_triton,
            
            mask_b_ptr = mask_b,
            mask_q_ptr = mask_q,
            mask_kv_ptr = mask_kv,
        
            query_begin_pos = query_begin_pos,
            query_end_pos = query_end_pos,
            
            N_Key_tokens = self.max_qk_pairs, ## Maximum number of kv 
            Sequence_Length = self.sequence_length, ## The sequence length or
            Head_dim = self.head_dim,
            N_head = self.n_heads,
            Batch_size = self.batch_size,
        )

        return v_out_triton


class TopKSparseAttention(nn.Module):
    def __init__(self,
                 embed_dim,
                 n_head_attention,
                 n_head_indexer,
                 head_dim_indexer,
                 top_k,
                ):
        super().__init__()

        self.n_head_attention = n_head_attention

        self.dense_mha = nn.MultiheadAttention(embed_dim, n_head_attention,batch_first=True)

        with torch.no_grad():
            eye = torch.eye(embed_dim)
            self.dense_mha.in_proj_weight.copy_(eye.repeat(3, 1))
            self.dense_mha.in_proj_bias.zero_()
        
            self.dense_mha.out_proj.weight.copy_(eye)
            self.dense_mha.out_proj.bias.zero_()

        self.dense_mha.requires_grad_(False)
        
        self.indexer = Indexer(
            hidden_dim = embed_dim,
            head_dim = head_dim_indexer,
            n_heads = n_head_indexer,
            TOP_K=top_k,
            topk_block_size=8192,
            USE_FLAT_MASK= True)

    def __get_softmax_score(self,
                            b,
                            q,
                            v,
                            score,
                            shape
                           ):

        sparse_scores = torch.sparse_coo_tensor(
        indices=torch.stack((b, q, v)),
        values=score.float(),
        size=shape,
        device=score.device,
        is_coalesced=True)

        scores = torch.sparse.log_softmax(sparse_scores, dim=-1)

        
        with torch.no_grad():
            q_idx = get_begin_idx(q)

        return scores.values(), len(q_idx)
    
    def forward(self,
                hidden,
                query,
                key,
                value,
                mask,
                layer):

        neighbour_mask = self.indexer.prepare_neighbour_mask(mask,layer)
        score,b,q,v =  self.indexer.prepare_score(hidden.detach(),neighbour_mask)
        neighbour_mask = neighbour_mask.unsqueeze(1)

        
        mha_mask = neighbour_mask.repeat(1,self.n_head_attention,1,1).reshape(-1,mask.shape[1],mask.shape[1])
        
        output_vec, attn_weight = self.dense_mha(query, key, value, attn_mask = ~mha_mask)

        output_vec = torch.nan_to_num(output_vec, nan=0.0)
        
        score, n_group = self.__get_softmax_score(b,q,v,score,neighbour_mask[:,0].shape)
        
        attn_weight = attn_weight[neighbour_mask.squeeze(1)]

        loss = F.kl_div(score,
                attn_weight.detach(),
                reduction="sum",
                log_target=False,)
        loss = loss/n_group
        return output_vec, loss