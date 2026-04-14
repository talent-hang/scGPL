import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from .dataset import Dataset, adj2saprse_tensor, collate_fn
from .model import scGPL
from .train_val import train, validate
from .losses import get_loss_function
import os
import csv
import random


def set_seed(seed=42):
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(data_dir, args):
    seed = getattr(args, 'seed', 42)
    set_seed(seed)
    print(f'Random seed set to: {seed}')

    expression_data_path = data_dir + '/BL--ExpressionData.csv'
    train_data_path = data_dir + '/Train_set.csv'
    val_data_path = data_dir + '/Validation_set.csv'
    test_data_path = data_dir + '/Test_set.csv'

    expression_data = np.array(pd.read_csv(expression_data_path, index_col=0, header=0))
    TF_path = data_dir + '/TF.csv'
    TF = torch.from_numpy(pd.read_csv(TF_path, index_col=0, header=0)['index'].values.astype(np.int64))

    train_data_df = pd.read_csv(train_data_path, index_col=0, header=0)
    train_pairs = train_data_df.iloc[:, :2].values
    train_gene_set = set(train_pairs.flatten())
    train_gene_indices = sorted(list(train_gene_set))

    train_expression_data = expression_data[train_gene_indices, :]
    standard = StandardScaler()
    standard.fit(train_expression_data.T)

    scaled_df = standard.transform(expression_data.T)
    expression_data = scaled_df.T
    expression_data_shape = expression_data.shape

    subgraph_hops = getattr(args, 'subgraph_hops', 2)
    max_subgraph_size = getattr(args, 'max_subgraph_size', 50)
    subgraph_strategy = getattr(args, 'subgraph_strategy', 'priority')

    train_dataset = Dataset(train_data_path, expression_data,
                           subgraph_hops=subgraph_hops, max_subgraph_size=max_subgraph_size,
                           subgraph_strategy=subgraph_strategy)
    train_adj_dict = train_dataset.gene_adj_dict
    train_global_edge_index = train_dataset.global_edge_index_cpu

    val_dataset = Dataset(val_data_path, expression_data,
                         subgraph_hops=subgraph_hops, max_subgraph_size=max_subgraph_size,
                         subgraph_strategy=subgraph_strategy,
                         gene_adj_dict=train_adj_dict,
                         global_edge_index=train_global_edge_index)
    test_dataset = Dataset(test_data_path, expression_data,
                          subgraph_hops=subgraph_hops, max_subgraph_size=max_subgraph_size,
                          subgraph_strategy=subgraph_strategy,
                          gene_adj_dict=train_adj_dict,
                          global_edge_index=train_global_edge_index)

    adj = train_dataset.Adj_Generate(TF, loop=False)
    adj = adj2saprse_tensor(adj).coalesce()

    Batch_size = args.batch_size
    Embed_size = args.embed_size
    Num_layers = args.num_layers
    Num_head = args.num_head
    LR = args.lr
    EPOCHS = args.epochs
    step_size = args.step_size
    gamma = args.gamma
    global schedulerflag
    schedulerflag = args.scheduler_flag

    torch.multiprocessing.set_start_method('spawn', force=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=Batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=8,
                                               collate_fn=collate_fn,
                                               persistent_workers=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=Batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=8,
                                             collate_fn=collate_fn,
                                             persistent_workers=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=Batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=8,
                                              collate_fn=collate_fn,
                                              persistent_workers=True)

    use_subgraph = getattr(args, 'use_subgraph', True)
    use_globalgraph = getattr(args, 'use_globalgraph', True)
    subgraph_gnn_layers = getattr(args, 'subgraph_gnn_layers', 2)
    globalgraph_gnn_layers = getattr(args, 'globalgraph_gnn_layers', 2)
    gnn_hidden = getattr(args, 'gnn_hidden', 256)
    gnn_type = getattr(args, 'gnn_type', 'gat')
    gnn_heads = getattr(args, 'gnn_heads', 4)
    gnn_dropout = getattr(args, 'gnn_dropout', 0.2)
    T = scGPL(expression_data_shape, Embed_size, Num_layers, Num_head,
                subgraph_gnn_layers=subgraph_gnn_layers, gnn_hidden=gnn_hidden, use_subgraph=use_subgraph,
                use_globalgraph=use_globalgraph, globalgraph_gnn_layers=globalgraph_gnn_layers,
                gnn_type=gnn_type, gnn_heads=gnn_heads, gnn_dropout=gnn_dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = T.to(device)
    optimizer = torch.optim.Adam(T.parameters(), lr=LR, weight_decay=getattr(args, 'weight_decay', 1e-5))
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    loss_type = getattr(args, 'loss_type', 'bce')
    loss_func = get_loss_function(loss_type=loss_type)

    print("current use_subgraph :",use_subgraph)
    print("current use_globalgraph :",use_globalgraph)
    print("current subgraph_strategy :",subgraph_strategy)
    if use_globalgraph:
        print("current globalgraph_gnn_layers :",globalgraph_gnn_layers)
    print("current subgraph_gnn_layers :",subgraph_gnn_layers)
    print("current gnn_hidden :",gnn_hidden)
    print("current gnn_type :",gnn_type)
    print("current gnn_heads :",gnn_heads)
    print("current gnn_dropout :",gnn_dropout)
    print("current loss function :",loss_type)

    best_val_auc = 0.0
    best_val_aupr = 0.0
    all_val_aucs = []
    all_val_auprs = []

    patience = args.patience
    epochs_no_improve = 0
    result_root = 'yourpath/result'
    model_dir = os.path.join(result_root, 'models')
    log_dir = os.path.join(result_root, 'logs')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    last3 = os.path.normpath(data_dir).split(os.sep)[-3:]
    model_name = '_'.join(last3) + '.pth'
    model_save_path = os.path.join(model_dir, model_name)
    best_model_wts = T.state_dict()

    for epoch in range(1, EPOCHS + 1):
        train(T, train_loader, loss_func, optimizer, epoch, scheduler, args)
        AUC_val, AUPR_val, _ = validate(T, val_loader, loss_func)

        if AUC_val > best_val_auc:
            best_val_auc = AUC_val
            epochs_no_improve = 0
            best_model_wts = T.state_dict()
        else:
            epochs_no_improve += 1
        if AUPR_val > best_val_aupr:
            best_val_aupr = AUPR_val

        all_val_aucs.append(AUC_val)
        all_val_auprs.append(AUPR_val)

        print('-' * 100)
        print(f'| end of epoch {epoch:3d} | valid AUROC {AUC_val:8.3f} | valid AUPRC {AUPR_val:8.3f}')
        print(f'| Current Best | valid AUROC {best_val_auc:8.3f} | valid AUPRC {best_val_aupr:8.3f}')

        if epoch % 5 == 0 or epoch == EPOCHS:
            AUC_test_epoch, AUPR_test_epoch, _ = validate(T, test_loader, loss_func)
            print('| epoch {:3d} | test AUROC {:8.3f} | test AUPRC {:8.3f}'.format(
                epoch, AUC_test_epoch, AUPR_test_epoch))

        if args.scheduler_flag:
            scheduler.step()

        print('-' * 100)

        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {patience} epochs with no improvement.')
            break

    avg_val_auc = sum(all_val_aucs) / len(all_val_aucs)
    avg_val_aupr = sum(all_val_auprs) / len(all_val_auprs)

    T.load_state_dict(best_model_wts)
    AUC_test, AUPR_test, EPR_test = validate(T, test_loader, loss_func)
    best_test_auc = AUC_test
    best_test_aupr = AUPR_test

    torch.save(best_model_wts, model_save_path)
    print(f"[Model Saved] 当前最优模型已保存到: {model_save_path}")

    print('\nFinal Results (Best on Validation → Test):')
    print(f'Average val AUROC: {avg_val_auc:.3f} | Best val AUROC: {best_val_auc:.3f}')
    print(f'Average val AUPRC: {avg_val_aupr:.3f} | Best val AUPRC: {best_val_aupr:.3f}')
    print(f'Test AUROC (best val model): {best_test_auc:.4f}')
    print(f'Test AUPRC (best val model): {best_test_aupr:.4f}')

    result_file = os.path.join(log_dir, f"{model_name}.log")
    with open(result_file, 'a') as f:
        f.write(f'==== Results for {data_dir} ====\n')
        f.write('\n=== Training Parameters ===\n')
        param_keys = [
            'batch_size', 'embed_size', 'num_layers', 'num_head', 'lr', 'epochs',
            'step_size', 'gamma', 'scheduler_flag', 'patience', 'loss_type',
            'focal_weight', 'bce_weight', 'consistency_weight', 'focal_alpha',
            'focal_gamma', 'label_smoothing', 'weight_decay',
            'use_subgraph', 'use_globalgraph', 'subgraph_strategy', 'subgraph_gnn_layers', 'globalgraph_gnn_layers',
            'gnn_hidden', 'subgraph_hops', 'max_subgraph_size', 'gnn_type', 'gnn_heads', 'gnn_dropout',
        ]
        for key in param_keys:
            if hasattr(args, key):
                value = getattr(args, key)
                f.write(f'{key}: {value}\n')
        f.write('\n=== Final Results ===\n')
        f.write(f'Average val AUROC: {avg_val_auc:.3f} | Best val AUROC: {best_val_auc:.3f}\n')
        f.write(f'Average val AUPRC: {avg_val_aupr:.3f} | Best val AUPRC: {best_val_aupr:.3f}\n')
        f.write(f'Test AUROC (best val model): {best_test_auc:.4f}\n')
        f.write(f'Test AUPRC (best val model): {best_test_aupr:.4f}\n')
        f.write(f'Final test EPR (top-100): {EPR_test:.4f}\n')
        f.write('\n')
    print(f"Results appended to {result_file}")

    # 写入汇总 summary.csv
    summary_path = os.path.join(result_root, 'summary.csv')
    headers = [
        'network_type', 'cell_type', 'tf_config', 'seed',
        'test_auroc', 'test_auprc', 'test_epr',
        'use_subgraph', 'use_globalgraph', 'subgraph_strategy', 'gnn_type', 'subgraph_gnn_layers', 'globalgraph_gnn_layers',
        'gnn_hidden', 'gnn_heads', 'gnn_dropout', 'subgraph_hops', 'max_subgraph_size'
    ]
    network_type, cell_type, tf_config = last3 if len(last3) == 3 else ('NA', 'NA', 'NA')
    # 获取种子信息（优先从args获取，其次从环境变量）
    seed = getattr(args, 'seed', os.environ.get('SEED', 'unknown'))
    row = [
        network_type, cell_type, tf_config, seed,
        f'{best_test_auc:.4f}', f'{best_test_aupr:.4f}', f'{EPR_test:.4f}',
        getattr(args, 'use_subgraph', True),
        getattr(args, 'use_globalgraph', True),
        getattr(args, 'subgraph_strategy', 'priority'),
        getattr(args, 'gnn_type', 'gat'),
        getattr(args, 'subgraph_gnn_layers', 2),
        getattr(args, 'globalgraph_gnn_layers', 2),
        getattr(args, 'gnn_hidden', 256),
        getattr(args, 'gnn_heads', 4),
        getattr(args, 'gnn_dropout', 0.2),
        getattr(args, 'subgraph_hops', 2),
        getattr(args, 'max_subgraph_size', 50),
    ]
    file_exists = os.path.exists(summary_path)
    with open(summary_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(row)
    print(f"Summary appended to {summary_path}")
