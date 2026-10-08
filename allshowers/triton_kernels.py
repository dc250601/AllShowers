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

    v = tl.load(
        v_ptr + offsets,
        mask=mask,
        other=0,
    )

    s = tl.load(
        s_ptr + offsets,
        mask=mask,
        other=-float("inf"),
    )

    sorted_scores = tl.sort(s, descending=True)

    k = tl.minimum(size,TOP_K)


    topk_score = tl.max(
        tl.where(
            ranks == (k - 1),
            sorted_scores,
            -float("inf"),
        ),
        axis=0,
    )

    topk_mask = mask & (s >= topk_score)
    
    tl.store(
    output_ptr + offsets,
    v,
    mask=topk_mask,)


## Mask as a dimension of m,3 where 3--> B,Q,V
## output ptr ---> B,L.L

@triton.jit
def triton_custom_mask(
    mask_stream_ptr,
    output_ptr,
    n_elements: tl.constexpr,
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
    