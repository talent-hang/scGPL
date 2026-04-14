import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from gnn_module import build_gnn_layer


class scGPL(nn.Module):
    def __init__(self, expression_data_shape, embed_size, num_layers, num_head,
                 subgraph_gnn_layers, gnn_hidden, use_subgraph, gnn_type, gnn_heads, gnn_dropout,
                 use_globalgraph, globalgraph_gnn_layers):
        super(scGPL, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embed_size = embed_size
        self.use_subgraph = use_subgraph
        self.use_globalgraph = use_globalgraph
        self.globalgraph_gnn_layers = globalgraph_gnn_layers

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_head, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.position_embedding = nn.Embedding(2, embed_size)

        self.encoder512 = nn.Linear(expression_data_shape[1], 512)
        self.encoder768 = nn.Linear(512, embed_size)

        self.flatten = nn.Flatten()
        
        if use_subgraph:
            self.gnn_input_dim = expression_data_shape[1]
            self.subgraph_gnn_layers = subgraph_gnn_layers
            self.gnn_hidden = gnn_hidden

            self.gnn_layers_list = nn.ModuleList()
            self.gnn_layers_list.append(build_gnn_layer(gnn_type, self.gnn_input_dim, gnn_hidden,
                                                        num_heads=gnn_heads, dropout=gnn_dropout))
            for _ in range(subgraph_gnn_layers - 1):
                self.gnn_layers_list.append(build_gnn_layer(gnn_type, gnn_hidden, gnn_hidden,
                                                            num_heads=gnn_heads, dropout=gnn_dropout))
            
            self.subgraph_proj = nn.Linear(gnn_hidden, embed_size)
            self.subgraph_dropout = nn.Dropout(p=0.3)  # 较高的 dropout 率以增强正则化

        if use_globalgraph and use_subgraph:
            self.global_gnn_input_dim = expression_data_shape[1]
            self.global_gnn_layers_list = nn.ModuleList()
            self.global_gnn_layers_list.append(build_gnn_layer(gnn_type, self.global_gnn_input_dim, gnn_hidden,
                                                              num_heads=gnn_heads, dropout=gnn_dropout))
            for _ in range(globalgraph_gnn_layers - 1):
                self.global_gnn_layers_list.append(build_gnn_layer(gnn_type, gnn_hidden, gnn_hidden,
                                                                  num_heads=gnn_heads, dropout=gnn_dropout))

            self.global_proj = nn.Linear(gnn_hidden, embed_size)
            self.global_dropout = nn.Dropout(p=0.3)

        fusion_input_dim = embed_size * 2  
        if use_subgraph:
            fusion_input_dim += embed_size * 2 
        if use_globalgraph:
            fusion_input_dim += embed_size * 2 
        
        self.linear1024 = nn.Linear(fusion_input_dim, 1024)
        self.layernorm1024 = nn.LayerNorm(1024)

        self.linear512 = nn.Linear(1024, 512)
        self.layernorm512 = nn.LayerNorm(512)

        self.linear256 = nn.Linear(512, 256)
        self.layernorm256 = nn.LayerNorm(256)

        self.linear2 = nn.Linear(256, 1)
        self.actf = nn.PReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.pool = nn.AvgPool1d(kernel_size=4, stride=4)

    def _process_subgraph(self, batched_subgraph_expr, batched_edge_index, batch_indices, tf_indices, target_indices):
        """批处理所有子图，一次性在GPU上计算"""
        batched_subgraph_expr = batched_subgraph_expr.to(self.device)
        batched_edge_index = batched_edge_index.to(self.device)
        tf_indices = tf_indices.to(self.device)
        target_indices = target_indices.to(self.device)
        
        if batched_subgraph_expr.dim() == 1:
            batched_subgraph_expr = batched_subgraph_expr.unsqueeze(0)
        elif batched_subgraph_expr.dim() > 2:
            batched_subgraph_expr = batched_subgraph_expr.view(batched_subgraph_expr.size(0), -1)
        
        if batched_subgraph_expr.size(-1) != self.gnn_input_dim:
            raise RuntimeError(
                f"Subgraph expression dimension mismatch: "
                f"expected {self.gnn_input_dim}, got {batched_subgraph_expr.size(-1)}. "
                f"Subgraph shape: {batched_subgraph_expr.shape}"
            )
        
        layer_outputs = []
        h = batched_subgraph_expr
        layer_outputs.append(h)  # layer 0: 原始特征

        for i, gnn_layer in enumerate(self.gnn_layers_list):
            h = gnn_layer(h, batched_edge_index)
            if i > 0 and h.shape == batched_subgraph_expr.shape:
                h = h + batched_subgraph_expr  # 残差连接
            h = F.layer_norm(h, h.shape[1:])  # 层归一化
            h = F.leaky_relu(h)
            layer_outputs.append(h)  # 保存每层输出

        layer_tf_features = []
        layer_target_features = []

        for layer_h in layer_outputs:
            tf_feat = layer_h[tf_indices]  
            target_feat = layer_h[target_indices]
            layer_tf_features.append(tf_feat)
            layer_target_features.append(target_feat)

        
        for i in range(1, len(layer_tf_features)):  
            layer_tf_features[i] = self.subgraph_proj(layer_tf_features[i])
            layer_target_features[i] = self.subgraph_proj(layer_target_features[i])

            # 对GNN特征添加dropout正则化
            layer_tf_features[i] = self.subgraph_dropout(layer_tf_features[i])
            layer_target_features[i] = self.subgraph_dropout(layer_target_features[i])

        return layer_tf_features, layer_target_features

    def _process_global_gnn(self, expression_data, global_edge_index, tf_indices, target_indices):
        """全局GNN分支：基于全局邻接矩阵学习基因嵌入"""
        # 数据移到GPU
        expression_data = expression_data.to(self.device)
        global_edge_index = global_edge_index.to(self.device)
        tf_indices = tf_indices.to(self.device)
        target_indices = target_indices.to(self.device)

        
        layer_outputs = []
        h = expression_data
        layer_outputs.append(h)  # layer 0: 原始特征

        for i, gnn_layer in enumerate(self.global_gnn_layers_list):
            h = gnn_layer(h, global_edge_index)
            # 残差连接和归一化
            if i > 0 and h.shape == expression_data.shape:
                h = h + expression_data
            h = F.layer_norm(h, h.shape[1:])
            h = F.leaky_relu(h)
            layer_outputs.append(h)

    
        layer_tf_features = []
        layer_target_features = []

        for layer_h in layer_outputs:
            tf_feat = layer_h[tf_indices]
            target_feat = layer_h[target_indices]
            layer_tf_features.append(tf_feat)
            layer_target_features.append(target_feat)


        for i in range(1, len(layer_tf_features)):
            layer_tf_features[i] = self.global_proj(layer_tf_features[i])
            layer_target_features[i] = self.global_proj(layer_target_features[i])
           
            layer_tf_features[i] = self.global_dropout(layer_tf_features[i])
            layer_target_features[i] = self.global_dropout(layer_target_features[i])

        return layer_tf_features, layer_target_features

    def forward(self, main_pair, main_expr, batched_subgraph_expr, batched_edge_index,
                batch_indices, tf_idx, target_idx, global_expr_data=None, global_edge_index=None, return_penultimate=False):
        bs = main_expr.shape[0]

        position = torch.Tensor([0, 1] * bs).reshape(bs, -1).to(torch.int32).to(self.device)
        p_e = self.position_embedding(position)

        out_expr_e = self.encoder512(main_expr)
        out_expr_e = F.leaky_relu(self.encoder768(out_expr_e))

        transformer_input = out_expr_e + p_e
        transformer_output = self.transformer_encoder(transformer_input)
        transformer_output_flat = self.flatten(transformer_output)  # [bs, embed_size * 2]

      
        fusion_components = []

        # 1. 基础Transformer特征（始终包含）
        base_transformer = transformer_output_flat  # [bs, embed_size * 2]
        fusion_components.append(base_transformer)

        # 2. Subgraph GNN分支特征
        if self.use_subgraph:
            # 获取各层GNN特征
            layer_tf_features, layer_target_features = self._process_subgraph(
                batched_subgraph_expr, batched_edge_index, batch_indices, tf_idx, target_idx
            )

            # 提取子图增强特征
            if len(layer_tf_features) > 1:  # 有GNN层输出
                subgraph_tf = layer_tf_features[1]      # Layer 1特征
                subgraph_target = layer_target_features[1]
                subgraph_features = torch.cat([subgraph_tf, subgraph_target], dim=1)  # [bs, embed_size*2]

                # 归一化
                subgraph_features = F.layer_norm(subgraph_features, subgraph_features.shape[1:])
                fusion_components.append(subgraph_features)

        # 3. 全局GNN分支特征
        if self.use_globalgraph and global_expr_data is not None and global_edge_index is not None:
            # 获取全局GNN特征
            global_tf_features, global_target_features = self._process_global_gnn(
                global_expr_data, global_edge_index, main_pair[:, 0], main_pair[:, 1]
            )

           
            if len(global_tf_features) > 1:  
                global_tf = global_tf_features[1]      
                global_target = global_target_features[1]
                global_features = torch.cat([global_tf, global_target], dim=1) 
                
                global_features = F.layer_norm(global_features, global_features.shape[1:])
                fusion_components.append(global_features)

        
        if len(fusion_components) > 1:
            fusion_vector = torch.cat(fusion_components, dim=1)
        else:
            fusion_vector = fusion_components[0]

        out = self.linear1024(fusion_vector)
        
        out = self.layernorm1024(out)
        out = self.dropout(out)
        out = self.actf(out)

        r = out.unsqueeze(1)
        r = self.pool(r)
        r = r.squeeze(1)

        out = self.linear512(out)
        out = self.layernorm512(out)
        out = self.dropout(out)
        out = self.actf(out)

        penultimate = out.clone()

        out = self.linear256(out) + r
        out = self.layernorm256(out)
        out = self.dropout(out)
        out = self.actf(out)

        outs = self.linear2(out)
        outs = nn.Sigmoid()(outs)

        if return_penultimate:
            return penultimate, outs
        else:
            return outs

