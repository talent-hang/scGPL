import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 默认损失函数配置
DEFAULT_LOSS_CONFIG = {
    'focal': {
        'alpha': 1.0,
        'gamma': 2.0,
        'reduction': 'mean'
    },
    'weighted_bce': {
        'pos_weight': 5.75
    },
    'combined': {
        'focal_weight': 0.7,
        'bce_weight': 0.3,
        'focal_alpha': 1.0,
        'focal_gamma': 2.0,
        'label_smoothing': 0.1
    },
    'label_smoothing': {
        'smoothing': 0.1
    }
}


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if inputs.dim() == 2 and inputs.size(1) == 1:
            inputs = inputs.squeeze(1)
        if targets.dim() == 2 and targets.size(1) == 1:
            targets = targets.squeeze(1)
            
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingBCELoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingBCELoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        if inputs.dim() == 2 and inputs.size(1) == 1:
            inputs = inputs.squeeze(1)
        if targets.dim() == 2 and targets.size(1) == 1:
            targets = targets.squeeze(1)
            
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy(inputs, targets)


class CombinedLoss(nn.Module):
    def __init__(self, focal_weight=0.7, bce_weight=0.3, focal_alpha=1, focal_gamma=2, label_smoothing=0.1):
        super(CombinedLoss, self).__init__()
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight
        
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.bce_loss = LabelSmoothingBCELoss(smoothing=label_smoothing)

    def forward(self, inputs, targets):
        if inputs.dim() == 2 and inputs.size(1) == 1:
            inputs = inputs.squeeze(1)
        if targets.dim() == 2 and targets.size(1) == 1:
            targets = targets.squeeze(1)
            
        focal_loss = self.focal_loss(inputs, targets)
        bce_loss = self.bce_loss(inputs, targets)
        
        total_loss = self.focal_weight * focal_loss + self.bce_weight * bce_loss
        
        return total_loss, {
            'focal_loss': focal_loss.item(),
            'bce_loss': bce_loss.item(),
            'consistency_loss': 0,
            'total_loss': total_loss.item()
        }


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=5.75):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        if inputs.dim() == 2 and inputs.size(1) == 1:
            inputs = inputs.squeeze(1)
        if targets.dim() == 2 and targets.size(1) == 1:
            targets = targets.squeeze(1)
            
        pos_mask = (targets == 1).float()
        neg_mask = (targets == 0).float()
        
        pos_count = pos_mask.sum()
        neg_count = neg_mask.sum()
        total_count = pos_count + neg_count
        
        if pos_count > 0 and neg_count > 0:
            pos_weight = neg_count / pos_count * self.pos_weight
            neg_weight = 1.0
        else:
            pos_weight = 1.0
            neg_weight = 1.0
        
        weights = pos_mask * pos_weight + neg_mask * neg_weight
        
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        weighted_loss = (bce_loss * weights).mean()
        
        return weighted_loss


def get_loss_function(loss_type='bce', **kwargs):
    """获取损失函数，使用默认配置或用户提供的kwargs"""
    if loss_type == 'focal':
        config = DEFAULT_LOSS_CONFIG.get('focal', {})
        config.update({k: v for k, v in kwargs.items() if k in ['alpha', 'gamma', 'reduction']})
        return FocalLoss(**config)
    elif loss_type == 'weighted_bce':
        config = DEFAULT_LOSS_CONFIG.get('weighted_bce', {})
        config.update({k: v for k, v in kwargs.items() if k in ['pos_weight']})
        return WeightedBCELoss(**config)
    elif loss_type == 'label_smoothing':
        config = DEFAULT_LOSS_CONFIG.get('label_smoothing', {})
        config.update({k: v for k, v in kwargs.items() if k in ['smoothing']})
        return LabelSmoothingBCELoss(**config)
    elif loss_type == 'combined':
        config = DEFAULT_LOSS_CONFIG.get('combined', {})
        config.update({k: v for k, v in kwargs.items() 
                      if k in ['focal_weight', 'bce_weight', 'focal_alpha', 'focal_gamma', 'label_smoothing']})
        return CombinedLoss(**config)
    else:  # 'bce'
        return nn.BCELoss() 