import os
import sys
import argparse

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
for path in (ROOT_DIR, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.main import main


def parse_args():
    parser = argparse.ArgumentParser(
        description='scGPL: Single-cell Universal Gene Prediction Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ========== 数据相关 ==========
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--data_dir', type=str,
                           default='yourpath/data/Dataspilt/Specific/hESC/TFs_1000',
                           help='Path to the dataset directory')

    # ========== 模型架构 ==========
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--embed_size', type=int, default=1024, help='Embedding dimension')
    model_group.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    model_group.add_argument('--num_head', type=int, default=8, help='Number of attention heads')

    # ========== 共享GNN参数 ==========
    shared_group = parser.add_argument_group('Shared GNN Parameters')
    shared_group.add_argument('--gnn_type', type=str, default='gat',
                             choices=['gat', 'gcn', 'graphsage', 'gin'],
                             help='GNN layer type (shared by subgraph and globalgraph branches)')
    shared_group.add_argument('--gnn_hidden', type=int, default=128,
                             help='GNN hidden dimension (shared by subgraph and globalgraph branches)')
    shared_group.add_argument('--gnn_heads', type=int, default=4,
                             help='Number of attention heads for GAT (shared, ignored for other GNN types)')
    shared_group.add_argument('--gnn_dropout', type=float, default=0.8,
                             help='GNN dropout rate (shared by subgraph and globalgraph branches)')

    # ========== 子图分支参数 ==========
    subgraph_group = parser.add_argument_group('Subgraph Branch Parameters')
    subgraph_group.add_argument('--use_subgraph', type=bool, default=True,
                               help='Enable subgraph GNN branch')
    subgraph_group.add_argument('--subgraph_strategy', type=str, default='priority',
                               choices=['bfs', 'priority', 'ppr'],
                               help='Subgraph extraction strategy: bfs (breadth-first), priority (importance-based), ppr (personalized pagerank)')
    subgraph_group.add_argument('--subgraph_gnn_layers', type=int, default=2,
                               help='Number of GNN layers for subgraph branch')
    subgraph_group.add_argument('--subgraph_hops', type=int, default=2,
                               help='Number of hops for subgraph extraction')
    subgraph_group.add_argument('--max_subgraph_size', type=int, default=100,
                               help='Maximum number of nodes per subgraph')

    # ========== 全局图分支参数 ==========
    global_group = parser.add_argument_group('Global Graph Branch Parameters')
    global_group.add_argument('--use_globalgraph', type=bool, default=True,
                             help='Enable global graph GNN branch based on full gene network')
    global_group.add_argument('--globalgraph_gnn_layers', type=int, default=2,
                             help='Number of GNN layers for global graph branch')

    # ========== 训练超参数 ==========
    train_group = parser.add_argument_group('Training Hyperparameters')
    train_group.add_argument('--batch_size', type=int, default=256, help='Batch size')
    train_group.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    train_group.add_argument('--epochs', type=int, default=200, help='Maximum number of epochs')
    train_group.add_argument('--weight_decay', type=float, default=1e-5,
                           help='Weight decay (L2 regularization coefficient)')
    train_group.add_argument('--patience', type=int, default=8,
                           help='Early stopping patience (epochs without improvement)')

    # ========== 学习率调度 ==========
    scheduler_group = parser.add_argument_group('Learning Rate Scheduler')
    scheduler_group.add_argument('--scheduler_flag', type=bool, default=True,
                               help='Enable learning rate scheduler')
    scheduler_group.add_argument('--step_size', type=int, default=10,
                               help='Step size for StepLR scheduler (epochs)')
    scheduler_group.add_argument('--gamma', type=float, default=0.999,
                               help='Multiplicative factor for StepLR scheduler')

    # ========== 损失函数 ==========
    loss_group = parser.add_argument_group('Loss Function')
    loss_group.add_argument('--loss_type', type=str, default='bce',
                           choices=['bce', 'focal', 'weighted_bce', 'combined'],
                           help='Loss function type')

    # ========== 其他 ==========
    other_group = parser.add_argument_group('Other')
    other_group.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print('data_dir:', args.data_dir)
    main(args.data_dir, args)
    print(f'Training or Evaluation on {args.data_dir} finished.')
