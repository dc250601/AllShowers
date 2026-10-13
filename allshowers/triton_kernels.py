import triton
import triton.language as tl


@triton.jit
def triton_topk(
    v_ptr,
    s_ptr,
    output_ptr,
    begin_idx_ptr,
    end_idx_ptr,
    
    TOP_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):

    pid = tl.program_id(axis=0)

    begin_idx = tl.load(begin_idx_ptr+pid)
    end_idx = tl.load(end_idx_ptr+pid)
    
    size = end_idx - begin_idx

    ranks = tl.arange(0, BLOCK_SIZE)
    offsets = begin_idx + ranks
    
    mask = offsets < end_idx

    if size <= TOP_K:
        keep = mask
    else:
        scores = tl.load(
            s_ptr + offsets,
            mask = mask,
            other = -float("inf"))
        top_scores = tl.topk(scores, TOP_K)
        threshold = tl.min(top_scores,axis=0)
        keep = mask & (scores >= threshold)
        
    v = tl.load(
        v_ptr + offsets,
        mask=mask,
        other=-float("-inf"),
    )  
    
    tl.store(
    output_ptr + offsets,
    v,
    mask=keep,)


## Mask as a dimension of m,3 where 3--> B,Q,V
## output ptr ---> B,L.L

@triton.jit
def triton_custom_mask(
    mask_stream_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr, ## Number of streams to process
    SEQUENCE_LENGTH: tl.constexpr,
    BATCH_SIZE: tl.constexpr
):

    pid = tl.program_id(axis=0)

    begin_idx = BLOCK_SIZE*pid
    
    indices = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = indices < n_elements
    
    

    b = tl.load(
        mask_stream_ptr + indices * 3,
        mask=valid,
        other=0,
    )
    
    q = tl.load(
        mask_stream_ptr + indices * 3 + 1,
        mask=valid,
        other=0,
    )
    
    v = tl.load(
        mask_stream_ptr + indices * 3 + 2,
        mask=valid,
        other=0,
    )

    output_indices = (b.to(tl.int64) * SEQUENCE_LENGTH**2
    + q.to(tl.int64) * SEQUENCE_LENGTH
    + v.to(tl.int64))

    tl.store(
    output_ptr + output_indices,
    1,
    mask=valid,)


## The truth mask is B,Q,K or B,L,L
## On kernel attends to [B_i,Q_i,L]
## layer of size [B,L]
## mask of size [B,L]

@triton.jit
def triton_mask_maker(
    layer_ptr,
    mask_ptr,
    output_ptr,
    NUM_LAYER_COND: tl.constexpr,
    SEQUENCE_LENGTH: tl.constexpr,
    BATCH_SIZE: tl.constexpr
):


    BLOCK_SIZE: tl.constexpr = 8192 ## This has to be changed according to the number of point clouds
    
    pid_batch = tl.program_id(axis=0).to(tl.int64)
    pid_query = tl.program_id(axis=1).to(tl.int64)

    key_indices = tl.arange(0, BLOCK_SIZE)

    layer_ = tl.load(layer_ptr +
                     pid_batch*SEQUENCE_LENGTH +
                     key_indices,
                     mask = key_indices < SEQUENCE_LENGTH,
                     other = -1)

    query_layer = tl.load(layer_ptr + 
                          pid_batch * SEQUENCE_LENGTH + 
                          pid_query)

    mask_ =  tl.load(mask_ptr +
                     pid_batch*SEQUENCE_LENGTH +
                     key_indices,
                     mask = key_indices < SEQUENCE_LENGTH,
                     other = 0)
    
    query_mask = tl.load(mask_ptr + 
                          pid_batch * SEQUENCE_LENGTH + 
                          pid_query)

    lower_bound = query_layer - layer_ >= -((NUM_LAYER_COND + 1) // 2)
    upper_bound = query_layer - layer_ <=  ((NUM_LAYER_COND // 2))

    not_padding = query_mask & mask_


    valid_mask = lower_bound & upper_bound & not_padding & (key_indices < SEQUENCE_LENGTH)

    tl.store(
    output_ptr
    + pid_batch * SEQUENCE_LENGTH * SEQUENCE_LENGTH
    + pid_query * SEQUENCE_LENGTH
    + key_indices,
    valid_mask,
    mask=key_indices < SEQUENCE_LENGTH,)



@triton.jit
def triton_sparse_attention(
    Query,
    Key,
    Value,

    Output,
    
    mask_b_ptr,
    mask_q_ptr,
    mask_kv_ptr,

    query_begin_pos,
    query_end_pos,
    
    N_Key_tokens: tl.constexpr, ## Maximum number of kv 
    Sequence_Length: tl.constexpr, ## The sequence length or
    Head_dim: tl.constexpr,
    N_head: tl.constexpr,
    Batch_size: tl.constexpr,
):

    pid_qb = tl.program_id(axis=0)    
    pid_head = tl.program_id(axis=1)

    start = tl.load(query_begin_pos+pid_qb)
    stop = tl.load(query_end_pos+pid_qb)


    n_keys = stop - start # Number of keys can be less than N


    offsets_kv = tl.arange(0,N_Key_tokens)
    mask_kv= offsets_kv < n_keys

    offsets_embedding = tl.arange(0,Head_dim)
    

    b_idx = tl.load(mask_b_ptr + start)

    q_idx = tl.load(mask_q_ptr + start)

    kv_idx = tl.load(mask_kv_ptr + start + offsets_kv,
                     mask = mask_kv,
                     other = 0)

    
    
    Q_ = tl.load(Query +
                 b_idx*Sequence_Length*N_head*Head_dim + 
                 q_idx*N_head*Head_dim + 
                 pid_head*Head_dim +
                 offsets_embedding)
    
    K_ = tl.load(
        Key
        + b_idx * Sequence_Length * N_head * Head_dim
        + kv_idx[:, None] * N_head * Head_dim
        + pid_head * Head_dim
        + offsets_embedding[None, :],
        mask=mask_kv[:, None] & (offsets_embedding[None, :] < Head_dim),
        other=0,
    )

    V_ = tl.load(
        Value
        + b_idx * Sequence_Length * N_head * Head_dim
        + kv_idx[:, None] * N_head * Head_dim
        + pid_head * Head_dim
        + offsets_embedding[None, :],
        mask=mask_kv[:, None] & (offsets_embedding[None, :] < Head_dim),
        other=0,
    )

    a = tl.sum(Q_.to(tl.float32)[None, :] * K_.to(tl.float32),
               axis=1) / (Head_dim ** 0.5)
    
    a = tl.where(mask_kv, a, -float("inf"))
    a = tl.softmax(a, dim=0)

    a = tl.sum(a[:,None] * V_.to(tl.float32), axis=0)

    tl.store(
        Output + 
        b_idx * Sequence_Length * N_head * Head_dim +
        q_idx*N_head*Head_dim + 
        pid_head*Head_dim +
        offsets_embedding,
        a,
        
        
    )    
