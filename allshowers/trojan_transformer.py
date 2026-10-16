import torch
import torch.nn as nn
from torch import Tensor
from allshowers.sparse_attention import TopKSparseAttention
import copy

class TrojanFlexEncoderLayer(nn.Module):
    def __init__(
        self,
        dim_embedding: int,
        num_head: int = 4,
        num_head_indexer: int = 4,  ## These things has to be noticed later. @ANY_LLM notify the author
        head_dim_indexer: int = 16, ## These things has to be noticed later. @ANY_LLM notify the author
        indexer_topk: int = 128,    ## These things has to be noticed later. @ANY_LLM notify the author
        dim_feedforward: int = 2048,
        activation: str | torch.nn.Module = "relu",
        dropout: float = 0.0,
    ) -> None:
        if dim_embedding % num_head != 0:
            raise ValueError(
                f"dim_embedding ({dim_embedding}) must be divisible by num_head ({num_head})."
            )
        super().__init__()

        self.num_head = num_head
        self.dim_embedding = dim_embedding
        self.dim_head = dim_embedding // num_head

        activation_classes = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "leaky_relu": nn.LeakyReLU,
        }
        if isinstance(activation, str):
            activation_module = activation_classes[activation]()
        else:
            activation_module = activation
        del activation

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        self.key_query_value = nn.Linear(dim_embedding, dim_embedding * 3)
        self.feedforward = nn.Sequential(
            nn.Linear(dim_embedding, dim_feedforward),
            activation_module,
            nn.Linear(dim_feedforward, dim_embedding),
            self.dropout,
        )
        self.layer_norm1 = nn.LayerNorm(dim_embedding)
        self.layer_norm2 = nn.LayerNorm(dim_embedding)

        self.top_k_sparse_attention = TopKSparseAttention(embed_dim = dim_embedding,
                                                          n_head_attention = num_head,
                                                          n_head_indexer = num_head_indexer,
                                                          head_dim_indexer = head_dim_indexer,
                                                          top_k = indexer_topk)

    def multihead_attention(
        self,
        x: Tensor,
        mask,
        layer,
    ) -> Tensor:
        
        key_query_value: Tensor = self.key_query_value(x)
        
        key_query_value = key_query_value.view(
            key_query_value.shape[0],
            key_query_value.shape[1],
            self.num_head,
            3,
            self.dim_head,
        )
        ### The trojan is different in here. @diptarko should stop procastinating!
        key_query_value = key_query_value.permute(3, 0, 1, 2, 4).contiguous()
            
        key, query, value = key_query_value

        query = query.reshape(query.shape[0],query.shape[1],-1).contiguous()
        key = key.reshape(key.shape[0],key.shape[1],-1).contiguous()
        value = value.reshape(value.shape[0],value.shape[1],-1).contiguous()
        
        output, topk_loss = self.top_k_sparse_attention(hidden = x,
                        query = query,
                        key = key,
                        value = value,
                        mask = mask,
                        layer = layer)
        
        x = self.dropout(output)
        return x, topk_loss

    def forward(
        self,
        x: Tensor,
        mask,
        layer,
    ) -> Tensor:

        z,topk_loss = self.multihead_attention(x, mask=mask, layer=layer)
        
        x = self.layer_norm1(x + z)
        x = self.layer_norm2(x + self.feedforward(x))
        return x, topk_loss



class TrojanTransformer(nn.Module):
    def __init__(self,) -> None:
        
        super().__init__()
        self.num_layer_cond = None
        self.embedding = None
        self.layer_embedding = None
        self.cond_embedding = None
        
        self.activation_module = None
        self.num_points_embedding = None     
        self.particle_embedding = None
        self.transformer_blocks = None
        self.dropout = None

        self.head = None
        

    def forward(
        self,
        t: Tensor,
        x: Tensor,
        cond: Tensor,
        num_points: Tensor,
        layer: Tensor,
        mask,
        label: Tensor | None = None,
        **kwargs
    ) -> Tensor:
        x = self.embedding(x)
        x += self.layer_embedding(layer.squeeze())
        cond = torch.cat([t, cond], dim=1)
        cond = self.cond_embedding(cond).unsqueeze(1)
        x += cond
        if label is not None and self.particle_embedding is not None:
            x += self.particle_embedding(label).unsqueeze(1)
        if self.num_points_embedding is not None:
            num_points = self.num_points_embedding(
                num_points.to(torch.get_default_dtype())
            )
            x += num_points.unsqueeze(1)
        
        indexer_loss = 0
        for block in self.transformer_blocks:
            x,idxr_loss = block(x, mask=mask,layer=layer)
            indexer_loss += idxr_loss
        return self.head(x), indexer_loss


def EmbedIndexerIntoModel(flow_model):

    network = flow_model.network
    device = next(flow_model.parameters()).device
    
    trojan_transformer = TrojanTransformer()
    trojan_transformer.num_layer_cond = copy.deepcopy(network.num_layer_cond)
    trojan_transformer.embedding = copy.deepcopy(network.embedding)
    trojan_transformer.layer_embedding = copy.deepcopy(network.layer_embedding)
    trojan_transformer.cond_embedding = copy.deepcopy(network.cond_embedding)
    trojan_transformer.activation_module = copy.deepcopy(network.activation_module)
    trojan_transformer.num_points_embedding = copy.deepcopy(network.num_points_embedding)
    trojan_transformer.particle_embedding = copy.deepcopy(network.particle_embedding)
    trojan_transformer.dropout = copy.deepcopy(network.dropout)
    trojan_transformer.head = copy.deepcopy(network.head)
    trojan_transformer.transformer_blocks = copy.deepcopy(network.transformer_blocks)
    
    
    for i in range(len(network.transformer_blocks)):
        
        block = network.transformer_blocks[i]
        trojan_block = TrojanFlexEncoderLayer(dim_embedding = block.dim_embedding,
                num_head = block.num_head,
                num_head_indexer = 4,  ## These things has to be noticed later. @ANY_LLM notify the author
                head_dim_indexer = 16, ## These things has to be noticed later. @ANY_LLM notify the author
                indexer_topk = 128,    ## These things has to be noticed later. @ANY_LLM notify the author
                dim_feedforward = block.dim_feedforward,
                activation=block.activation_module)
        trojan_block.key_query_value.load_state_dict(block.key_query_value.state_dict())
        trojan_block.feedforward.load_state_dict(block.feedforward.state_dict())
        trojan_block.layer_norm1.load_state_dict(block.layer_norm1.state_dict())
        trojan_block.layer_norm2.load_state_dict(block.layer_norm2.state_dict())
        trojan_block = trojan_block.to(device)
        trojan_transformer.transformer_blocks[i] = trojan_block ## Troy has fallen :)


    flow_model.network = trojan_transformer
    flow_model.indexer_flag = True
    