import torch
import warnings
from .triton_kernels import triton_sparse_attention

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