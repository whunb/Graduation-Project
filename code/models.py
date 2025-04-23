import torch.nn as nn
import torch.nn.functional as F
import torch
from dgllife.model.gnn import GCN
import RetNet
from RetNet import RetNet
from einops import reduce

class DynamicWeightFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.attention = nn.Sequential(
            nn.Linear(dim * 2, dim),  # 输入是两种特征拼接
            nn.ReLU(),
            nn.Linear(dim, 2),        # 输出两个模态的权重
            nn.Softmax(dim=1)         # 对权重做归一化
        )
    
    def forward(self, feat1, feat2):
        # feat1和feat2的形状: (batch_size, dim)
        combined = torch.cat([feat1, feat2], dim=1)  # (batch_size, dim*2)
        weights = self.attention(combined)           # (batch_size, 2)
        fused = weights[:, 0].unsqueeze(1) * feat1 + weights[:, 1].unsqueeze(1) * feat2
        return fused  # (batch_size, dim)



def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss

def mean_square_error(v_d, v_s):
    loss_fct = torch.nn.MSELoss()
    loss = loss_fct(v_d, v_s)
    return loss

def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent

class MGNDTI(nn.Module):
    def __init__(self, **config):
        super(MGNDTI, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        drug_layers = config["DRUG"]["LAYERS"]
        drug_num_head = config["DRUG"]["NUM_HEAD"]
        drug_padding = config["DRUG"]["PADDING"]

        protein_layers = config["PROTEIN"]["LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        protein_num_head = config['PROTEIN']['NUM_HEAD']
        protein_padding = config["PROTEIN"]["PADDING"]

        mgn_emb_dim = config["MGN"]["EMBEDDING_DIM"]

        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        out_binary = config["DECODER"]["BINARY"]

        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding,
                                           hidden_feats=drug_hidden_feats)
        self.smiles_extractor = MolecularRetNet(embedding_dim=drug_embedding,
                                                num_head=drug_num_head, layers=drug_layers, padding=drug_padding)
        self.protein_extractor = ProteinRetNet(embedding_dim=protein_emb_dim,
                                               num_head=protein_num_head, layers=protein_layers, padding=protein_padding)
        #Multimodal Gating Network
        self.multi_gating_network = MultimodalGatingNetwork(mgn_emb_dim)
        # #CrossModalAttentionNetwork
        # self.multi_gating_network = CrossModalAttentionNetwork(mgn_emb_dim)

        #MLPDecoder
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def forward(self, smi_d, bg_d, v_p, mode="train"):
        #Drug Encoder
        v_d = self.drug_extractor(bg_d)
        v_s = self.smiles_extractor(smi_d)
        #Protein Encoder
        v_p = self.protein_extractor(v_p)
        #Multimodal Gating Network
        f, v_d, v_s, v_p = self.multi_gating_network(v_d, v_s, v_p)
        #Decoder
        score = self.mlp_classifier(f)
        if mode == "train":
            return v_d, v_s, v_p, f, score
        elif mode == "eval":
            return v_d, v_s, v_p, score, None

class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats

class ProteinRetNet(nn.Module):
    def __init__(self, embedding_dim, num_head, layers, padding=True):
        super(ProteinRetNet, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)

        self.retnet = RetNet(layers=layers, hidden_dim=embedding_dim,
                             ffn_size=embedding_dim // 2, heads=num_head, double_v_dim=False)

    def forward(self, v):
        v = self.embedding(v.long())
        v = self.retnet(F.relu(v))

        return v

class GLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GLU, self).__init__()
        self.W = nn.Linear(in_dim, out_dim)
        self.V = nn.Linear(in_dim, out_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        Y = self.W(X) * self.sigmoid(self.V(X))
        return Y

# # 跨模态注意力融合
# class CrossModalAttentionNetwork(nn.Module):
#     def __init__(self, dim):
#         super(CrossModalAttentionNetwork, self).__init__()
#         self.query_proj = nn.Linear(dim, dim)
#         self.key_proj = nn.Linear(dim, dim)
#         self.value_proj = nn.Linear(dim, dim)
#         self.softmax = nn.Softmax(dim=-1)
#         self.tanh = nn.Tanh()

#     def cross_attention(self, q, k, v):
#         # q, k, v: [B, L, D]
#         q = self.query_proj(q)
#         k = self.key_proj(k)
#         v = self.value_proj(v)
#         attn = torch.bmm(q, k.transpose(1, 2)) / (q.shape[-1] ** 0.5)
#         attn = self.softmax(attn)
#         output = torch.bmm(attn, v)
#         return output

#     def forward(self, g_feats, s_feats, p_feats):
#         # drug graph vs. smiles, protein vs. smiles
#         g2s = self.cross_attention(g_feats, s_feats, s_feats)  # drug graph attends to SMILES
#         p2s = self.cross_attention(p_feats, s_feats, s_feats)  # protein attends to SMILES

#         # Reduce by mean
#         v_d = g2s.mean(dim=1)
#         v_s = s_feats.mean(dim=1)
#         v_p = p2s.mean(dim=1)

#         # 交叉融合
#         f = self.tanh(torch.cat([v_d * v_p, v_s * v_p], dim=-1))
#         return f, v_d, v_s, v_p


class MultimodalGatingNetwork(nn.Module):
    def __init__(self, dim):
        super(MultimodalGatingNetwork, self).__init__()
        self.gated_g = GLU(dim, dim)
        self.gated_s = GLU(dim, dim)
        self.gated_p = GLU(dim, dim)
        self.dynamic_fuse = DynamicWeightFusion(dim)  # 新增动态融合模块
        self.tanh = nn.Tanh()

    def forward(self, mg, ms, mp):
        # mg, ms, mp 的形状: (batch_size, seq_len, dim)
        mg = self.gated_g(mg)
        ms = self.gated_s(ms)
        mp = self.gated_p(mp)
        
        # 最大池化得到全局特征
        v_d = reduce(mg, "b h w -> b w", 'max')  # (batch_size, dim)
        v_s = reduce(ms, "b h w -> b w", 'max')  # (batch_size, dim)
        v_p = reduce(mp, "b h w -> b w", 'max')  # (batch_size, dim)

        # 动态融合药物分子图和SMILES特征
        v_d_fused = self.dynamic_fuse(v_d, v_s)  # (batch_size, dim)
        
        # 与靶点特征交互
        v_dp = v_d_fused * v_p  # (batch_size, dim)
        v_sp = v_s * v_p        # (batch_size, dim)
        
        # 拼接最终特征
        f = self.tanh(torch.cat([v_dp, v_sp], dim=-1))  # (batch_size, dim*2)
        return f, v_d, v_s, v_p  # 保持返回格式兼容原代码

class MolecularRetNet(nn.Module):
    def __init__(self, embedding_dim, num_head, layers, padding=True):
        super(MolecularRetNet, self).__init__()
        if padding:
            self.embedding = nn.Embedding(65, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(65, embedding_dim)

        self.retnet = RetNet(layers=layers, hidden_dim=embedding_dim,
                             ffn_size=embedding_dim // 2, heads=num_head, double_v_dim=False)

    def forward(self, v):
        v = self.embedding(v.long())
        v = self.retnet(F.relu(v))
        return v



class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):#x.shpae[64, 256]
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x





