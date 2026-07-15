import torch
from utils import Evaluation
from losses import get_loss_function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, dataloader, loss_func, optimizer, epoch, scheduler, args):
    model.train()
    log_interval = 200
    total_loss = 0
    loss_components = {'focal_loss': 0, 'bce_loss': 0, 'consistency_loss': 0, 'total_loss': 0}

    for idx, (main_pair, main_expr, batched_subgraph_expr, batched_edge_index, batch_indices, tf_idx, target_idx, label, global_expr_data, global_edge_index) in enumerate(dataloader):
        main_pair = main_pair.to(device)
        main_expr = main_expr.to(torch.float32).to(device)
        labels = label.to(torch.float32).to(device)
        # 子图数据已经在collate_fn中转换为tensor，在model中移到GPU

        optimizer.zero_grad()

        predicted_label = model(main_pair, main_expr, batched_subgraph_expr, batched_edge_index,
                               batch_indices, tf_idx, target_idx, global_expr_data, global_edge_index)
        
        if predicted_label.dim() == 2 and predicted_label.size(1) == 1:
            predicted_label = predicted_label.squeeze(1)
        
        if hasattr(loss_func, 'forward') and hasattr(loss_func, 'focal_weight'):
            loss, components = loss_func(predicted_label, labels)
            for key in loss_components:
                loss_components[key] += components[key]
        else:
            loss = loss_func(predicted_label, labels)
            loss_components['total_loss'] += loss.item()
        
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        if idx % log_interval == 0 and idx > 0:
            AUROC, AUPRC = Evaluation(y_pred=predicted_label.detach(), y_true=labels)
            print('| epoch {:3d} | {:5d}/{:5d} batches | Train loss {:8.3f} | AUROC {:8.3f} | AUPRC {:8.3f}'.format(
                epoch, idx, len(dataloader), loss.item(), AUROC, AUPRC))
            
            if hasattr(loss_func, 'focal_weight'):
                print('  | Focal: {:.4f} | BCE: {:.4f} | Consistency: {:.4f}'.format(
                    components['focal_loss'], components['bce_loss'], components['consistency_loss']))

    avg_loss = total_loss / len(dataloader)
    print('| epoch {:3d} | average loss {:8.3f}'.format(epoch, avg_loss))
    
    if hasattr(loss_func, 'focal_weight'):
        for key in loss_components:
            loss_components[key] /= len(dataloader)
        print('  | Avg Focal: {:.4f} | Avg BCE: {:.4f} | Avg Consistency: {:.4f}'.format(
            loss_components['focal_loss'], loss_components['bce_loss'], loss_components['consistency_loss']))


def validate(model, dataloader, loss_func):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        pre = []
        lb = []

        for idx, (main_pair, main_expr, batched_subgraph_expr, batched_edge_index, batch_indices, tf_idx, target_idx, label, global_expr_data, global_edge_index) in enumerate(dataloader):
            main_pair = main_pair.to(device)
            main_expr = main_expr.to(torch.float32).to(device)
            labels = label.to(torch.float32).to(device).unsqueeze(1)
            # 子图数据已经在collate_fn中转换为tensor，在model中移到GPU

            predicted_label = model(main_pair, main_expr, batched_subgraph_expr, batched_edge_index,
                                   batch_indices, tf_idx, target_idx, global_expr_data, global_edge_index)

            if predicted_label.dim() == 2 and predicted_label.size(1) == 1:
                predicted_label = predicted_label.squeeze(1)
            if labels.dim() == 2 and labels.size(1) == 1:
                labels = labels.squeeze(1)

            if hasattr(loss_func, 'forward') and hasattr(loss_func, 'focal_weight'):
                loss, _ = loss_func(predicted_label, labels)
            else:
                loss = loss_func(predicted_label, labels)
            
            total_loss += loss.item()

            if predicted_label.dim() == 1:
                predicted_label = predicted_label.unsqueeze(1)
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)
                
            pre.extend(predicted_label.detach().cpu())
            lb.extend(labels.detach().cpu())

        pre = torch.vstack(pre)
        lb = torch.vstack(lb)

        AUROC, AUPRC = Evaluation(y_pred=pre, y_true=lb)
        # 计算 EPR（top-K 命中率相对基线提升倍数）
        probs = pre.flatten()
        labels_flat = lb.flatten()
        k = min(100, probs.numel())
        if k > 0 and labels_flat.sum() > 0:
            topk_vals, topk_idx = torch.topk(probs, k)
            hits = labels_flat[topk_idx].sum().item()
            precision_topk = hits / k
            base_rate = labels_flat.sum().item() / labels_flat.numel()
            epr = precision_topk / base_rate if base_rate > 0 else 0.0
        else:
            epr = 0.0

    print('| Validation | average loss {:8.3f} | AUROC {:8.3f} | AUPRC {:8.3f}'.format(
        total_loss / len(dataloader), AUROC, AUPRC))

    return AUROC, AUPRC, epr