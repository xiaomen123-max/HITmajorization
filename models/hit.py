import matplotlib.pyplot as plt 
import torch
import random
import torch.backends.cudnn as cudnn
import numpy as np 
import pickle
import argparse
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc as auc3
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold

from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class IntraMA(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_heads, num_hyperedge_types=4, num_node_types=2, dropout_rate=0.1, device='cpu'):
        super().__init__()
        print('-- Initializing the IntraMutualAttention....')
        self.device = device
        self.num_heads = num_heads
        
        # For hyperedges
        self.hyper_query_linears = nn.ModuleDict({
            str(i): nn.Linear(input_dim, embedding_dim * num_heads, bias=False) for i in [0, 2, 3, 4]
        })
        # Nodes will be the keys and values
        self.node_key_linears = nn.ModuleDict({
            str(i): nn.Linear(input_dim, embedding_dim * num_heads, bias=False) for i in [0, 1]
        })
        self.node_value_linears = nn.ModuleDict({
            str(i): nn.Linear(input_dim, embedding_dim * num_heads, bias=False) for i in [0, 1]
        })

        self.scale = embedding_dim ** 0.5
        self.to_out = nn.Linear(embedding_dim*num_heads+input_dim, input_dim)
        
        self.norm_edge = nn.LayerNorm(input_dim)
        self.norm_node = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.SiLU()

        # 新增相对位置编码层
        self.max_rel_pos = 10  # 最大相对位置距离
        self.rel_pos_emb = nn.Embedding(2 * self.max_rel_pos + 1, num_heads)
        self.rel_pos_emb = self.rel_pos_emb.to(device)
        
    def forward(self, hyperedge_features, node_features, incidence_matrix, node_types, hyperedge_types):
        device = hyperedge_features.device
        
        hyperedge_features = self.norm_edge(hyperedge_features)
        node_features = self.norm_node(node_features)
        
        # Using different matrices for hyperedges based on type
        q = torch.stack([self.hyper_query_linears[str(hyperedge_types[i])](hyperedge_features[i]) for i in range(hyperedge_features.shape[0])])
        
        # Using different matrices for nodes based on type
        k = torch.stack([self.node_key_linears[str(node_types[i])](node_features[i]) for i in range(node_features.shape[0])])
        v = torch.stack([self.node_value_linears[str(node_types[i])](node_features[i]) for i in range(node_features.shape[0])])
        
        q = rearrange(q, 'n (h d) -> h n d', h=self.num_heads)
        k = rearrange(k, 'n (h d) -> h n d', h=self.num_heads)
        v = rearrange(v, 'n (h d) -> h n d', h=self.num_heads)
        incidence_matrix = repeat(incidence_matrix, 'i j -> h j i', h=self.num_heads)

        scores = torch.einsum('hid,hjd->hij', q, k) / self.scale
        
        scores = scores.masked_fill(incidence_matrix == 0, float('-inf'))
        
         # 修正位置编码计算部分
        seq_len_q = q.size(1)  # 查询序列长度
        seq_len_k = k.size(1)  # 键序列长度
        rel_pos = torch.arange(seq_len_q, device=device).unsqueeze(1) - torch.arange(seq_len_k, device=device).unsqueeze(0)
        rel_pos = torch.clamp(rel_pos, -self.max_rel_pos, self.max_rel_pos) + self.max_rel_pos
        pos_embed = self.rel_pos_emb(rel_pos)  # [seq_len_q, seq_len_k, num_heads]
        pos_embed = pos_embed.permute(2, 0, 1)  # [num_heads, seq_len_q, seq_len_k]
        
        scores = scores + pos_embed
        
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        output = torch.einsum('hij,hjd->hid', weights, v)
        output = rearrange(output, 'h n d -> n (h d)', h=self.num_heads)
        output = self.act(output)
        
        final_output = torch.cat((hyperedge_features, output), dim=-1)
        final_output = self.to_out(final_output)

        return final_output, scores



    
class InterMA(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_heads, num_hyperedge_types=4, num_node_types=2, dropout_rate=0.1):
        super().__init__()
        print('-- Initializing the InterMutualAttention....')
        
        self.num_heads = num_heads
        
        # For nodes
        self.node_query_linears = nn.ModuleDict({
            str(i): nn.Linear(input_dim, embedding_dim * num_heads, bias=False) for i in [0, 1]
        })
        # Hyperedges will be the keys and values
        self.hyper_key_linears = nn.ModuleDict({
            str(i): nn.Linear(input_dim, embedding_dim * num_heads, bias=False) for i in [0, 2, 3, 4]
        })
        self.hyper_value_linears = nn.ModuleDict({
            str(i): nn.Linear(input_dim, embedding_dim * num_heads, bias=False) for i in [0, 2, 3, 4]
        })

        self.scale = embedding_dim ** 0.5
        self.to_out = nn.Linear(embedding_dim*num_heads+input_dim, input_dim)
        self.norm_edge = nn.LayerNorm(input_dim)
        self.norm_node = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.SiLU()
        
        # 新增相对位置编码层
        self.max_rel_pos = 10  # 最大相对位置距离
        self.rel_pos_emb = nn.Embedding(2 * self.max_rel_pos + 1, num_heads)


    def forward(self, intra_mutual_attention_output, node_features, incidence_matrix, node_types, hyperedge_types):
        device = intra_mutual_attention_output.device
        intra_mutual_attention_output = self.norm_edge(intra_mutual_attention_output)
        node_features = self.norm_node(node_features)
        
        # Using different matrices for nodes based on type
        q = torch.stack([self.node_query_linears[str(node_types[i])](node_features[i]) for i in range(node_features.shape[0])])
        
        # Using different matrices for hyperedges based on type
        k = torch.stack([self.hyper_key_linears[str(hyperedge_types[i])](intra_mutual_attention_output[i]) for i in range(intra_mutual_attention_output.shape[0])])
        v = torch.stack([self.hyper_value_linears[str(hyperedge_types[i])](intra_mutual_attention_output[i]) for i in range(intra_mutual_attention_output.shape[0])])
        
        q = rearrange(q, 'n (h d) -> h n d', h=self.num_heads)
        k = rearrange(k, 'n (h d) -> h n d', h=self.num_heads)
        v = rearrange(v, 'n (h d) -> h n d', h=self.num_heads)
        incidence_matrix = repeat(incidence_matrix, 'i j -> h i j', h=self.num_heads)

        scores = torch.einsum('hid,hjd->hij', q, k) / self.scale
        scores = scores.masked_fill(incidence_matrix == 0, float('-inf'))
        
        # 修正位置编码计算部分
        seq_len_q = q.size(1)  # 查询序列长度
        seq_len_k = k.size(1)  # 键序列长度
        rel_pos = torch.arange(seq_len_q, device=device).unsqueeze(1) - torch.arange(seq_len_k, device=device).unsqueeze(0)
        rel_pos = torch.clamp(rel_pos, -self.max_rel_pos, self.max_rel_pos) + self.max_rel_pos
        pos_embed = self.rel_pos_emb(rel_pos)  # [seq_len_q, seq_len_k, num_heads]
        pos_embed = pos_embed.permute(2, 0, 1)  # 
        
        scores = scores + pos_embed
        
        
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        output = torch.einsum('hij,hjd->hid', weights, v)
        output = rearrange(output, 'h n d -> n (h d)', h=self.num_heads)
        output = self.act(output)
        final_output = torch.cat((node_features, output), dim=-1)
        final_output = self.to_out(final_output)

        return final_output, scores

    
class MetaHGT(nn.Module) : 
    def __init__(self, input_dim, embedding_dim, num_heads) : 
        super().__init__()
        print('-- Initializing the Meta_HGT_Layer....')

        # Intra Mutual Attention
        self.Intra = IntraMA(input_dim, embedding_dim, num_heads)
        # Inter Mutual Attention 
        self.Inter = InterMA(input_dim, embedding_dim, num_heads)

    def forward(self, hyperedge_features, node_features, incidence_matrix, node_types, hyperedge_types) : 
        scores = {}
        updated_hyperedges, score = self.Intra(hyperedge_features, node_features, incidence_matrix, node_types, hyperedge_types)
        scores['Intra'] = score
        updated_node_embeddings, score = self.Inter(updated_hyperedges, node_features, incidence_matrix, node_types, hyperedge_types)
        scores['Inter'] = score

        return updated_hyperedges, updated_node_embeddings, scores
    
class Semantic_Attention(nn.Module) : 
    def __init__(
        self, 
        all_genes, 
        edge_list, 
        node_list, 
        genes_list, 
        input_dim, 
        num_heads, 
        device='cpu'
    ) :  
        super().__init__()
        print('-- Initializing the Semantic_Attention....')

        self.device = device
        self.embedding = torch.zeros(len(all_genes), input_dim).to(self.device)

        self.hyperedges = genes_list 
        self.nodes = node_list 

        self.weight = nn.Parameter(torch.randn(input_dim))
        nn.init.normal_(self.weight, mean=0, std=0.01)
        
    def forward(self, node, edge) :  
        emb1 = self.embedding.clone()
        emb1[self.nodes] = node
    
        emb2 = self.embedding.clone()
        emb2[self.hyperedges] = edge
        
        assert emb1.shape == emb2.shape

        repr_go = torch.mean(emb1, dim=0)
        repr_gene = torch.mean(emb2, dim=0)
        
        repr = torch.stack([repr_go, repr_gene], dim=1).to(self.device)
        scores = torch.matmul(self.weight, repr)
        attn = F.softmax(scores, dim=-1)

        out = (attn[0]*emb1) + (attn[1]*emb2)
        
        return out, scores
    

class HIT(nn.Module) : 
    def __init__(self, all_genes, all_diseases, input_dim, embedding_dim, num_heads, integrated_gene_go_hyperedge_list, integrated_gene_go_node_list, hpo_do_gene_all_keys, diseaseid_list_in_hpodogene, device='cpu') : 
        super().__init__()
        print('Initializing the HIT....')
        self.device = device

        self.lists = {
            'all_genes' : list(all_genes), # All Gene used's index
            'all_disease' : list(all_diseases), # All disease used's index
            
            'integrated_gene_go_hyperedge_list' : integrated_gene_go_hyperedge_list, # GO and Gene index list in Gene workflow (Hyperedge index)
            'genes_in_integrated_gene_go_hyperedge_list' : list(set(all_genes).intersection(set(integrated_gene_go_hyperedge_list))), # Gene's index list belongs to Gene-go hyperedge (Hyperedge index)
            'integrated_gene_go_node_list' : integrated_gene_go_node_list, # Gene index list in Gene workflow (Node index)
            
            'hpo_do_gene_all_keys' : hpo_do_gene_all_keys, # HPO, DO, Gene index list in Disease workflow (Hyperedge index)
            'diseaseid_list' : diseaseid_list_in_hpodogene, # Disease id list in hpo, do, gene hypergraph (Node index)
            
            'total_embedding_matrix' : list(range(62071)) # 0~66077 , the number of total embeddings (Gene, GO, Disease, HPO, DO)
        }

        self.embeddings = nn.Embedding(len(self.lists['total_embedding_matrix']), input_dim)
        
        # Layers for Gene
        self.metaHGT_g = MetaHGT(input_dim, embedding_dim, num_heads)

        # Gene semantic attention layer
        self.semantic = Semantic_Attention(all_genes, integrated_gene_go_hyperedge_list, integrated_gene_go_node_list, self.lists['genes_in_integrated_gene_go_hyperedge_list'], input_dim, num_heads, self.device)

        # Layers for disease
        self.metaHGT_d = MetaHGT(input_dim, embedding_dim, num_heads)

        types = []
        for i in range(62071) : 
            if i <= 21629 : 
                types.append(0) # gene
            elif (i > 21629) & (i <= 35833) : 
                types.append(2) # go
            elif (i > 35833) & (i <= 49237) : 
                types.append(1) # disease
            elif (i > 49237) & (i <= 55777) : 
                types.append(3) # hpo
            elif (i > 55777) & (i <= 62070) : 
                types.append(4) # do 
        
        self.types = torch.tensor(types)

        # 新增动态图学习参数
        self.dynamic_alpha = nn.Parameter(torch.tensor(0.5))  # 动态权重参数
        self.dynamic_beta = nn.Parameter(torch.tensor(0.5))   # 动态权重参数
        self.dynamic_gate = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def compute_dynamic_adjacency(self, static_adj, node_embeddings):
        """优化后的动态邻接矩阵计算，确保维度匹配"""
        with torch.cuda.amp.autocast():
            # 计算节点相似度
            sim_matrix = torch.matmul(node_embeddings, node_embeddings.t())
            
            # 确保相似度矩阵与静态邻接矩阵维度一致
            if sim_matrix.size() != static_adj.size():
                # 如果维度不匹配，调整相似度矩阵大小
                sim_matrix = F.interpolate(sim_matrix.unsqueeze(0).unsqueeze(0), 
                                         size=(static_adj.size(0), static_adj.size(1)),
                                         mode='bilinear').squeeze()
            
            # 只处理非零部分
            mask = static_adj != 0
            static_adj_values = static_adj[mask].view(-1, 1)
            sim_matrix_values = sim_matrix[mask].view(-1, 1)
            
            # 使用门控机制融合静态和动态关系
            gate_input = torch.cat([static_adj_values, sim_matrix_values], dim=-1)
            gate = self.dynamic_gate(gate_input)
            
            # 融合静态和动态邻接矩阵
            dynamic_values = gate * static_adj_values + (1 - gate) * sim_matrix_values
            
            # 重建稀疏矩阵，确保维度与原始矩阵一致
            dynamic_adj = torch.zeros_like(static_adj, dtype=torch.float32)
            dynamic_adj[mask] = dynamic_values.squeeze().float()
            
        return dynamic_adj
    
    def gen_geneflow_expression(self, embedding_matrix, gene_total_incidence_matrix) : 
        gene_node_feature_matrix = embedding_matrix[self.lists['integrated_gene_go_node_list']]
        gene_and_go_hyperedge_feature_matrix = embedding_matrix[self.lists['integrated_gene_go_hyperedge_list']]
        initial_gene_embedding = embedding_matrix[self.lists['all_genes']]

        ## node type, hyperedge type 
        gene_node_types = self.types[self.lists['integrated_gene_go_node_list']]
        gene_node_types = list(gene_node_types.detach().cpu().numpy())
        
        gene_hyperedge_types = self.types[self.lists['integrated_gene_go_hyperedge_list']]
        gene_hyperedge_types = list(gene_hyperedge_types.detach().cpu().numpy())

        # calculating gene expression using the extracted embeddings 
        gene_hyperedge_embeddings, gene_node_embeddings, scores_gene = self.metaHGT_g(gene_and_go_hyperedge_feature_matrix, gene_node_feature_matrix, gene_total_incidence_matrix, gene_node_types, gene_hyperedge_types)
        embedding_matrix[self.lists['integrated_gene_go_hyperedge_list']] = gene_hyperedge_embeddings 
        gene_hyperedge_embeddings = embedding_matrix[self.lists['genes_in_integrated_gene_go_hyperedge_list']] 
        # final gene expression (consider all relationships) 
        gene_expression, score = self.semantic(gene_node_embeddings, gene_hyperedge_embeddings) 
        scores_gene['semantic'] = score
        # Skip connection for gene final output 
        gene_expression = gene_expression + initial_gene_embedding
        return embedding_matrix, gene_expression, scores_gene

    def gen_diseaseflow_expression(self, embedding_matrix, gene_expression, disease_integrated_incidence) : 
        embedding_matrix[self.lists['all_genes']] = gene_expression 
        disease_integrated_node_feature_matrix = embedding_matrix[self.lists['diseaseid_list']] 
        disease_integrated_hyperedge_feature_matrix = embedding_matrix[self.lists['hpo_do_gene_all_keys']]
        initial_disease_embedding = embedding_matrix[self.lists['diseaseid_list']] 

        ## node type, hyperedg type 
        disease_node_types = self.types[self.lists['diseaseid_list']]
        disease_node_types = list(disease_node_types.detach().cpu().numpy())
        
        disease_hyperedge_types = self.types[self.lists['hpo_do_gene_all_keys']]
        disease_hyperedge_types = list(disease_hyperedge_types.detach().cpu().numpy())
        disease_hyperedge_embeddings, disease_node_embeddings, scores_disease = self.metaHGT_d(disease_integrated_hyperedge_feature_matrix, disease_integrated_node_feature_matrix, disease_integrated_incidence, disease_node_types, disease_hyperedge_types)
        # Skip connection for disease final output
        disease_expression = disease_node_embeddings + initial_disease_embedding
        return embedding_matrix, disease_hyperedge_embeddings, disease_expression, scores_disease
    

    def forward(self, gene_total_incidence_matrix, disease_integrated_incidence) : 
        scores_attention = {}
        embedding_matrix = self.embeddings(torch.tensor(self.lists['total_embedding_matrix']).to(self.device))
        
        gene_total_incidence_matrix = gene_total_incidence_matrix.float()
        disease_integrated_incidence = disease_integrated_incidence.float()
        
        # 计算动态图结构
        gene_dynamic_adj = self.compute_dynamic_adjacency(
            gene_total_incidence_matrix,
            embedding_matrix[self.lists['integrated_gene_go_node_list']]
        )
        
        disease_dynamic_adj = self.compute_dynamic_adjacency(
            disease_integrated_incidence,
            embedding_matrix[self.lists['diseaseid_list']]
        )
        
        # Calculate Updated total embedding matrix, update gene embeddings
        # embedding_matrix, gene_expression, scores = self.gen_geneflow_expression(embedding_matrix, gene_total_incidence_matrix) 
        # 使用动态图结构进行计算
        embedding_matrix, gene_expression, scores = self.gen_geneflow_expression(
            embedding_matrix, 
            self.dynamic_alpha * gene_total_incidence_matrix + 
            (1 - self.dynamic_alpha) * gene_dynamic_adj
        )
        scores_attention['gene'] = scores
        
        # Calcuate Updated total embedding matrix, updated hpo-do-gene embeddings, update disease embeddings
        embedding_matrix, disease_hyperedge_embeddings, disease_expression, scores = self.gen_diseaseflow_expression(
            embedding_matrix, 
            gene_expression,
            self.dynamic_beta * disease_integrated_incidence + 
            (1 - self.dynamic_beta) * disease_dynamic_adj
        )
        scores_attention['disease'] = scores
        
        embedding_matrix[self.lists['hpo_do_gene_all_keys']] = disease_hyperedge_embeddings
        embedding_matrix[self.lists['diseaseid_list']] = disease_expression

        return embedding_matrix, scores_attention
        
        


