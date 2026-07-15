from cProfile import label
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp

from collections import defaultdict
import heapq
import math


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, expression_data, subgraph_hops=2, max_subgraph_size=100,
                 gene_adj_dict=None, subgraph_strategy='priority', score_weights=(0.4, 0.6),
                 global_edge_index=None):
        data = pd.read_csv(data_path, index_col=0, header=0)
        train_data = pd.read_csv(data_path, index_col=0).values
        self.train_set = train_data
        self.dataset = np.array(data.iloc[:, :2])
        label = np.array(data.iloc[:, -1])
        self.label = np.eye(2)[label]
        self.label = label
        self.expression_data = expression_data
       
        self.num_gene = expression_data.shape[0]
        self.subgraph_hops = subgraph_hops
        self.max_subgraph_size = max_subgraph_size
        self.subgraph_strategy = subgraph_strategy
        self.score_weights = score_weights

        self.ppr_alpha = 0.85
        self.ppr_threshold = 0.9
        self.ppr_max_iter = 100

        self.pad_value = np.mean(expression_data, axis=0)

        if gene_adj_dict is not None:
            self.gene_adj_dict = {k: list(v) for k, v in gene_adj_dict.items()}
        else:
            self.gene_adj_dict = self._build_gene_adj_dict()

        degrees = [len(self.gene_adj_dict.get(i, [])) for i in range(self.num_gene)]
        self._max_degree = max(degrees) if degrees else 1

        expr = np.asarray(self.expression_data, dtype=np.float32)
        norms = np.linalg.norm(expr, axis=1)
        norms[norms == 0] = 1.0
        self._expr = expr
        self._expr_norms = norms
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self._expr_t = torch.from_numpy(expr).float().to(self.device)
            norms_t = torch.from_numpy(norms).float().to(self.device)
            self._expr_norms_t = norms_t
        except Exception:
            self._expr_t = None
            self._expr_norms_t = None

        self._precompute_gpu_adjacency()

        if global_edge_index is not None:
            self.global_edge_index_cpu = global_edge_index.clone() if hasattr(global_edge_index, 'clone') else global_edge_index
        else:
            self._build_global_adjacency_matrix()

    def _precompute_gpu_adjacency(self):
        try:
            row, col, data = [], [], []
            for i in range(self.num_gene):
                neighbors = self.gene_adj_dict.get(i, [])
                if neighbors:
                    degree = len(neighbors)
                    for j in neighbors:
                        row.append(i)
                        col.append(j)
                        data.append(1.0 / degree)

            if row:
                indices = torch.tensor([row, col], dtype=torch.long)
                values = torch.tensor(data, dtype=torch.float32)
                self.adj_matrix_cpu = torch.sparse_coo_tensor(indices, values, (self.num_gene, self.num_gene))
            else:
                self.adj_matrix_cpu = torch.sparse_coo_tensor(
                    torch.empty((2, 0), dtype=torch.long),
                    torch.empty(0, dtype=torch.float32),
                    (self.num_gene, self.num_gene)
                )

            degrees = torch.tensor([len(self.gene_adj_dict.get(i, [])) for i in range(self.num_gene)],
                                 dtype=torch.float32)
            self.node_degrees_cpu = degrees

        except Exception as e:
            print(f"Warning: Failed to precompute adjacency matrix: {e}")
            self.adj_matrix_cpu = None
            self.node_degrees_cpu = None

    def _build_global_adjacency_matrix(self):
        try:
            global_adj_dict = defaultdict(set)

            for tf, target, label in self.train_set:
                if label == 1:
                    global_adj_dict[tf].add(target)
                    global_adj_dict[target].add(tf)

            row, col = [], []
            for i in range(self.num_gene):
                neighbors = global_adj_dict.get(i, set())
                for j in neighbors:
                    row.append(i)
                    col.append(j)

            if row:
                indices = torch.tensor([row, col], dtype=torch.long)
                values = torch.ones(len(row), dtype=torch.float32)
                self.global_adj_matrix_cpu = torch.sparse_coo_tensor(indices, values, (self.num_gene, self.num_gene))

                self.global_edge_index_cpu = indices
            else:
                self.global_adj_matrix_cpu = torch.sparse_coo_tensor(
                    torch.empty((2, 0), dtype=torch.long),
                    torch.empty(0, dtype=torch.float32),
                    (self.num_gene, self.num_gene)
                )
                self.global_edge_index_cpu = torch.empty((2, 0), dtype=torch.long)

        except Exception as e:
            print(f"Warning: Failed to build global adjacency matrix: {e}")
            self.global_adj_matrix_cpu = None
            self.global_edge_index_cpu = None

    def _batch_extract_subgraphs_ppr(self, tf_indices, target_indices):
        if self.adj_matrix_cpu is None:
            return self._batch_extract_subgraphs_ppr_cpu(tf_indices, target_indices)

        batch_size = len(tf_indices)
        device = self.device

        adj_matrix = self.adj_matrix_cpu.to(device)

        ppr_scores = torch.zeros((batch_size, self.num_gene), dtype=torch.float32, device=device)

        for i, (tf, target) in enumerate(zip(tf_indices, target_indices)):
            ppr_scores[i, tf] = 0.5
            ppr_scores[i, target] = 0.5

        current_scores = ppr_scores.clone()

        for _ in range(self.ppr_max_iter):
            new_scores = torch.sparse.mm(adj_matrix.t(), current_scores.t()).t()

            new_scores = self.ppr_alpha * new_scores + (1 - self.ppr_alpha) * ppr_scores

            diff = torch.max(torch.abs(new_scores - current_scores))
            if diff < 1e-6:
                break

            current_scores = new_scores

        subgraph_nodes_list = []
        tf_local_indices = []
        target_local_indices = []

        for i, (tf, target) in enumerate(zip(tf_indices, target_indices)):
            scores = current_scores[i]

            sorted_indices = torch.argsort(scores, descending=True)
            selected_nodes = [tf, target]

            cumulative_prob = 1.0

            for node_idx in sorted_indices:
                node_idx = node_idx.item()
                if node_idx in [tf, target]:
                    continue
                if len(selected_nodes) >= self.max_subgraph_size:
                    break

                selected_nodes.append(node_idx)
                cumulative_prob += scores[node_idx].item()

                if cumulative_prob >= self.ppr_threshold:
                    break

            subgraph_nodes_list.append(selected_nodes)

            node_to_local_idx = {node: idx for idx, node in enumerate(selected_nodes)}
            tf_local_indices.append(node_to_local_idx[tf])
            target_local_indices.append(node_to_local_idx[target])

        return subgraph_nodes_list, tf_local_indices, target_local_indices

    def _batch_extract_subgraphs_ppr_cpu(self, tf_indices, target_indices):
        subgraph_nodes_list = []
        tf_local_indices = []
        target_local_indices = []

        for tf, target in zip(tf_indices, target_indices):
            nodes, _, tf_idx, target_idx = self._extract_subgraph_ppr(tf, target)
            subgraph_nodes_list.append(nodes)
            tf_local_indices.append(tf_idx)
            target_local_indices.append(target_idx)

        return subgraph_nodes_list, tf_local_indices, target_local_indices

    def batch_extract_subgraphs(self, tf_indices, target_indices):
        """统一的批量子图提取接口"""
        if self.subgraph_strategy == 'priority':
            return self._batch_extract_subgraphs_priority(tf_indices, target_indices)
        elif self.subgraph_strategy == 'ppr':
            return self._batch_extract_subgraphs_ppr(tf_indices, target_indices)
        else:  # default bfs
            return self._batch_extract_subgraphs_bfs(tf_indices, target_indices)

    def _batch_extract_subgraphs_bfs(self, tf_indices, target_indices):
        """批量提取BFS子图"""
        subgraph_nodes_list = []
        tf_local_indices = []
        target_local_indices = []

        for tf, target in zip(tf_indices, target_indices):
            nodes, _, tf_idx, target_idx = self._extract_subgraph_bfs(tf, target)
            subgraph_nodes_list.append(nodes)
            tf_local_indices.append(tf_idx)
            target_local_indices.append(target_idx)

        return subgraph_nodes_list, tf_local_indices, target_local_indices

    def _batch_extract_subgraphs_priority(self, tf_indices, target_indices):
        """批量提取Priority子图"""
        subgraph_nodes_list = []
        tf_local_indices = []
        target_local_indices = []

        for tf, target in zip(tf_indices, target_indices):
            nodes, _, tf_idx, target_idx = self._extract_subgraph_priority(tf, target)
            subgraph_nodes_list.append(nodes)
            tf_local_indices.append(tf_idx)
            target_local_indices.append(target_idx)

        return subgraph_nodes_list, tf_local_indices, target_local_indices

    def _extract_subgraph_bfs(self, tf, target):
        visited = set()
        nodes = set([tf, target])
        queue = [(tf, 0), (target, 0)]
        visited.add(tf)
        visited.add(target)

        while queue and len(nodes) < self.max_subgraph_size:
            node, hop = queue.pop(0)
            if hop >= self.subgraph_hops:
                continue

            for neighbor in self.gene_adj_dict.get(node, []):
                if neighbor not in visited and len(nodes) < self.max_subgraph_size:
                    visited.add(neighbor)
                    nodes.add(neighbor)
                    queue.append((neighbor, hop + 1))

        node_list = sorted(list(nodes))
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}

        edges = []
        for node in node_list:
            for neighbor in self.gene_adj_dict.get(node, []):
                if neighbor in node_to_idx:
                    edges.append((node_to_idx[node], node_to_idx[neighbor]))

        tf_idx = node_to_idx.get(tf, 0)
        target_idx = node_to_idx.get(target, 1 if len(node_list) > 1 else 0)

        return node_list, edges, tf_idx, target_idx

    def _build_gene_adj_dict(self):
        adj_dict = defaultdict(list)
        for pos in self.train_set:
            tf, target, label = pos
            if label == 1:
                adj_dict[tf].append(target)
                adj_dict[target].append(tf)
        return adj_dict

    def _extract_subgraph(self, tf, target):
        if self.subgraph_strategy == 'priority':
            return self._extract_subgraph_priority(tf, target)
        elif self.subgraph_strategy == 'ppr':
            return self._extract_subgraph_ppr(tf, target)
        visited = set()
        nodes = set([tf, target])
        queue = [(tf, 0), (target, 0)]
        visited.add(tf)
        visited.add(target)

        while queue and len(nodes) < self.max_subgraph_size:
            node, hop = queue.pop(0)
            if hop >= self.subgraph_hops:
                continue

            for neighbor in self.gene_adj_dict.get(node, []):
                if neighbor not in visited and len(nodes) < self.max_subgraph_size:
                    visited.add(neighbor)
                    nodes.add(neighbor)
                    queue.append((neighbor, hop + 1))

        node_list = sorted(list(nodes))
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}

        edges = []
        for node in node_list:
            for neighbor in self.gene_adj_dict.get(node, []):
                if neighbor in node_to_idx:
                    edges.append((node_to_idx[node], node_to_idx[neighbor]))

        tf_idx = node_to_idx.get(tf, 0)
        target_idx = node_to_idx.get(target, 1 if len(node_list) > 1 else 0)

        return node_list, edges, tf_idx, target_idx

    def _node_score(self, node, tf, target):
        deg = len(self.gene_adj_dict.get(node, []))
        deg_norm = deg / float(self._max_degree) if self._max_degree > 0 else 0.0
        expr_sim = 0.0
        try:
            if self._expr_t is not None and self._expr_norms_t is not None:
                vec = self._expr_t[node]
                tf_vec = self._expr_t[tf]
                target_vec = self._expr_t[target]
                vnorm = self._expr_norms_t[node]
                tf_norm = self._expr_norms_t[tf]
                target_norm = self._expr_norms_t[target]
                tf_sim = torch.dot(vec, tf_vec) / (vnorm * tf_norm)
                target_sim = torch.dot(vec, target_vec) / (vnorm * target_norm)
                expr_sim = float(torch.max(tf_sim, target_sim).item())
            else:
                vec = self._expr[node]
                vnorm = self._expr_norms[node]
                tf_vec = self._expr[tf]
                target_vec = self._expr[target]
                tf_sim = float(np.dot(vec, tf_vec) / (vnorm * self._expr_norms[tf]))
                target_sim = float(np.dot(vec, target_vec) / (vnorm * self._expr_norms[target]))
                expr_sim = max(tf_sim, target_sim)
        except Exception:
            expr_sim = 0.0

        w_deg, w_expr = self.score_weights
        score = w_deg * deg_norm + w_expr * expr_sim
        if not math.isfinite(score):
            score = 0.0
        return score

    def _extract_subgraph_priority(self, tf, target):
        """使用向量化的 priority-BFS（批量 top-k 扩展）提取子图以减少 Python 层面的循环开销"""
        visited = set([tf, target])
        nodes = set([tf, target])

        frontier = []
        for src in (tf, target):
            for nbr in self.gene_adj_dict.get(src, []):
                if nbr not in visited:
                    frontier.append(nbr)
        frontier = list(dict.fromkeys(frontier))

        while frontier and len(nodes) < self.max_subgraph_size:
            candidates = [n for n in frontier if n not in visited]
            if not candidates:
                break

            remaining = self.max_subgraph_size - len(nodes)

            scores = self._batch_node_scores(candidates, tf, target)

            k = min(len(candidates), remaining)
            if isinstance(scores, torch.Tensor):
                topk_vals, topk_idx = torch.topk(scores, k)
                top_indices = topk_idx.cpu().tolist()
            else:
                idxs = np.argsort(-scores)[:k]
                top_indices = idxs.tolist()

            selected = [candidates[i] for i in top_indices]

            for node in selected:
                visited.add(node)
                nodes.add(node)

            new_frontier = []
            for node in selected:
                for nbr in self.gene_adj_dict.get(node, []):
                    if nbr not in visited and nbr not in nodes:
                        new_frontier.append(nbr)

            frontier = list(dict.fromkeys(new_frontier))

        node_list = sorted(list(nodes))
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        edges = []
        for node in node_list:
            for neighbor in self.gene_adj_dict.get(node, []):
                if neighbor in node_to_idx:
                    edges.append((node_to_idx[node], node_to_idx[neighbor]))

        tf_idx = node_to_idx.get(tf, 0)
        target_idx = node_to_idx.get(target, 1 if len(node_list) > 1 else 0)
        return node_list, edges, tf_idx, target_idx

    def _extract_subgraph_ppr(self, tf, target):
        """使用 Personalized PageRank 提取子图，按 PPR 值降序选择节点"""
        ppr_scores = np.zeros(self.num_gene, dtype=np.float32)
        ppr_scores[tf] = 0.5
        ppr_scores[target] = 0.5

        row, col, data = [], [], []
        for i in range(self.num_gene):
            neighbors = self.gene_adj_dict.get(i, [])
            if neighbors:
                degree = len(neighbors)
                for j in neighbors:
                    row.append(i)
                    col.append(j)
                    data.append(1.0 / degree)

        if not row:
            node_list = sorted([tf, target])
            node_to_idx = {node: idx for idx, node in enumerate(node_list)}
            edges = []
            tf_idx = node_to_idx.get(tf, 0)
            target_idx = node_to_idx.get(target, 1 if len(node_list) > 1 else 0)
            return node_list, edges, tf_idx, target_idx

        import scipy.sparse as sp
        adj_matrix = sp.csr_matrix((data, (row, col)), shape=(self.num_gene, self.num_gene))

        teleport = ppr_scores.copy()
        current_scores = ppr_scores.copy()

        for _ in range(self.ppr_max_iter):
            new_scores = self.ppr_alpha * adj_matrix.dot(current_scores) + (1 - self.ppr_alpha) * teleport

            if np.max(np.abs(new_scores - current_scores)) < 1e-6:
                break

            current_scores = new_scores

        ppr_scores = current_scores

        all_nodes = [(i, ppr_scores[i]) for i in range(self.num_gene) if i not in [tf, target]]
        all_nodes.sort(key=lambda x: x[1], reverse=True)

        selected_nodes = [tf, target]
        cumulative_prob = 1.0

        for node_idx, score in all_nodes:
            if len(selected_nodes) >= self.max_subgraph_size:
                break

            selected_nodes.append(node_idx)
            cumulative_prob += score

            if cumulative_prob >= self.ppr_threshold:
                break

        node_list = sorted(selected_nodes)
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}

        edges = []
        for node in node_list:
            for neighbor in self.gene_adj_dict.get(node, []):
                if neighbor in node_to_idx:
                    edges.append((node_to_idx[node], node_to_idx[neighbor]))

        tf_idx = node_to_idx.get(tf, 0)
        target_idx = node_to_idx.get(target, 1 if len(node_list) > 1 else 0)

        return node_list, edges, tf_idx, target_idx

    def _batch_node_scores(self, node_list, tf, target):
        try:
            if self._expr_t is not None and self._expr_norms_t is not None:
                nodes_idx = torch.tensor(node_list, dtype=torch.long, device=self.device)
                vecs = self._expr_t[nodes_idx]
                tf_vec = self._expr_t[tf].unsqueeze(0)
                target_vec = self._expr_t[target].unsqueeze(0)
                vnorms = self._expr_norms_t[nodes_idx]
                tf_norm = self._expr_norms_t[tf]
                target_norm = self._expr_norms_t[target]

                tf_dots = (vecs * tf_vec).sum(dim=1) / (vnorms * tf_norm + 1e-8)
                target_dots = (vecs * target_vec).sum(dim=1) / (vnorms * target_norm + 1e-8)
                expr_sim = torch.max(tf_dots, target_dots)

                degs = torch.tensor([len(self.gene_adj_dict.get(n, [])) for n in node_list],
                                    dtype=torch.float32, device=self.device)
                deg_norm = degs / float(self._max_degree) if self._max_degree > 0 else degs * 0.0

                w_deg, w_expr = self.score_weights
                scores = w_deg * deg_norm + w_expr * expr_sim
                return scores
        except Exception:
            pass

        degs = np.array([len(self.gene_adj_dict.get(n, [])) for n in node_list], dtype=np.float32)
        deg_norm = degs / float(self._max_degree) if self._max_degree > 0 else degs * 0.0
        expr_sim_list = []
        for node in node_list:
            vec = self._expr[node]
            vnorm = self._expr_norms[node]
            tf_vec = self._expr[tf]
            target_vec = self._expr[target]
            tf_sim = float(np.dot(vec, tf_vec) / (vnorm * self._expr_norms[tf] + 1e-8))
            target_sim = float(np.dot(vec, target_vec) / (vnorm * self._expr_norms[target] + 1e-8))
            expr_sim_list.append(max(tf_sim, target_sim))
        expr_sim = np.array(expr_sim_list, dtype=np.float32)
        w_deg, w_expr = self.score_weights
        scores = w_deg * deg_norm + w_expr * expr_sim
        return scores

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        main_pair = self.dataset[i]
        main_label = self.label[i]
        tf, target = main_pair

        g1_expr = np.expand_dims(self.expression_data[tf], axis=0)
        g2_expr = np.expand_dims(self.expression_data[target], axis=0)
        main_expr = np.concatenate((g1_expr, g2_expr), axis=0)

        return (main_pair, main_expr, main_label, self)

    def Adj_Generate(self, TF_set, direction=False, loop=False):
        adj = sp.dok_matrix((self.num_gene, self.num_gene), dtype=np.float32)

        for pos in self.train_set:
            tf = pos[0]
            target = pos[1]

            if direction == False:
                if pos[-1] == 1:
                    adj[tf, target] = 1.0
                    adj[target, tf] = 1.0
            else:
                if pos[-1] == 1:
                    adj[tf, target] = 1.0
                    if target in TF_set:
                        adj[target, tf] = 1.0

        if loop:
            adj = adj + sp.identity(self.num_gene)

        adj = adj.todok()
        return adj

def collate_fn(batch):
    """自定义collate函数，使用批量子图提取优化GPU计算"""
    # 获取dataset对象（假设所有样本来自同一个dataset）
    dataset = batch[0][-1] if len(batch[0]) == 4 else None

    main_pairs = []
    main_exprs = []
    labels = []

    tf_indices = []
    target_indices = []

    for item in batch:
        if len(item) == 4:
            main_pair, main_expr, main_label, _ = item
        else:
            main_pair, main_expr, main_label = item[:3]

        tf, target = main_pair
        main_pairs.append(main_pair)
        main_exprs.append(main_expr)
        labels.append(main_label)
        tf_indices.append(tf)
        target_indices.append(target)

    main_pairs = torch.LongTensor(np.array(main_pairs))
    main_exprs = torch.FloatTensor(np.array(main_exprs))
    labels = torch.LongTensor(np.array(labels))

    if dataset is not None:
        subgraph_nodes_list, tf_local_indices, target_local_indices = dataset.batch_extract_subgraphs(
            tf_indices, target_indices)

        subgraph_exprs_list = []
        edge_indices_list = []
        batch_indices = []
        global_tf_indices = []
        global_target_indices = []

        node_offset = 0
        for i, (subgraph_nodes, tf_local_idx, target_local_idx) in enumerate(
            zip(subgraph_nodes_list, tf_local_indices, target_local_indices)):

            subgraph_expr = dataset.expression_data[subgraph_nodes].astype(np.float32)
            if subgraph_expr.ndim == 1:
                subgraph_expr = subgraph_expr.reshape(1, -1)
            subgraph_exprs_list.append(subgraph_expr)

            node_to_local_idx = {node: idx for idx, node in enumerate(subgraph_nodes)}
            edges = []
            for node in subgraph_nodes:
                for neighbor in dataset.gene_adj_dict.get(node, []):
                    if neighbor in node_to_local_idx:
                        edges.append((node_to_local_idx[node], node_to_local_idx[neighbor]))

            if edges:
                edge_index = np.array(edges, dtype=np.int64).T
                adjusted_edge_index = edge_index.copy()
                adjusted_edge_index[0] += node_offset
                adjusted_edge_index[1] += node_offset
                edge_indices_list.append(adjusted_edge_index)
            else:
                edge_indices_list.append(np.array([[], []], dtype=np.int64))

            global_tf_indices.append(tf_local_idx + node_offset)
            global_target_indices.append(target_local_idx + node_offset)

            batch_indices.extend([i] * len(subgraph_nodes))

            node_offset += len(subgraph_nodes)

        batched_subgraph_expr = np.vstack(subgraph_exprs_list)

        if edge_indices_list and any(ei.size > 0 for ei in edge_indices_list):
            valid_edges = [ei for ei in edge_indices_list if ei.size > 0]
            if valid_edges:
                batched_edge_index = np.concatenate(valid_edges, axis=1)
            else:
                batched_edge_index = np.array([[], []], dtype=np.int64)
        else:
            batched_edge_index = np.array([[], []], dtype=np.int64)

        batched_subgraph_expr = torch.FloatTensor(batched_subgraph_expr)
        batched_edge_index = torch.LongTensor(batched_edge_index) if batched_edge_index.size > 0 else torch.empty((2, 0), dtype=torch.long)
        batch_indices = torch.LongTensor(np.array(batch_indices))
        tf_indices = torch.LongTensor(np.array(global_tf_indices))
        target_indices = torch.LongTensor(np.array(global_target_indices))

        global_expr_data = torch.FloatTensor(dataset.expression_data) if dataset else torch.empty((0, 0), dtype=torch.float32)
        global_edge_index = dataset.global_edge_index_cpu if dataset and hasattr(dataset, 'global_edge_index_cpu') else None

    else:
        batched_subgraph_expr = torch.empty((0, dataset.expression_data.shape[1] if dataset else 0), dtype=torch.float32)
        batched_edge_index = torch.empty((2, 0), dtype=torch.long)
        batch_indices = torch.empty(0, dtype=torch.long)
        tf_indices = torch.zeros(len(batch), dtype=torch.long)
        target_indices = torch.zeros(len(batch), dtype=torch.long)
        global_expr_data = torch.empty((0, 0), dtype=torch.float32)
        global_edge_index = None

    return (main_pairs, main_exprs, batched_subgraph_expr, batched_edge_index,
            batch_indices, tf_indices, target_indices, labels, global_expr_data, global_edge_index)


def adj2saprse_tensor(adj):
    coo = adj.tocoo()
    i = torch.stack([torch.LongTensor(coo.row), torch.LongTensor(coo.col)])
    v = torch.FloatTensor(coo.data)
    return torch.sparse_coo_tensor(i, v, coo.shape)
