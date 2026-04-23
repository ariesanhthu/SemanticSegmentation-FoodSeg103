# File: loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=255, background_id=103, ce_weight=1.0, dice_weight=1.0, bg_weight=0.1):
        super().__init__()
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.background_id = background_id
        
        # Tạo trọng số cho Cross Entropy: Nền = 0.1, Các class khác = 1.0
        weight = torch.ones(num_classes)
        # Vì FoodSeg103 có background_id là 103
        if background_id < num_classes:
            weight[background_id] = bg_weight 
            
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, logits, targets):
        # 1. Tính Cross Entropy
        ce_loss = self.ce(logits, targets)
        
        # 2. Tính Dice Loss
        probas = F.softmax(logits, dim=1)
        num_classes = logits.size(1)
        
        # Tránh lỗi one_hot khi gặp ignore_index (255)
        valid_targets = targets.clone()
        valid_targets[valid_targets == self.ignore_index] = 0 # Đổi tạm thành 0 để không bị vỡ one_hot
        
        targets_1hot = F.one_hot(valid_targets, num_classes=num_classes).permute(0, 3, 1, 2).to(logits.dtype)
        
        # Áp dụng mask để loại bỏ hoàn toàn ignore_index khỏi quá trình tính Dice
        valid_mask = (targets != self.ignore_index).unsqueeze(1).float()
        probas = probas * valid_mask
        targets_1hot = targets_1hot * valid_mask
        
        dims = (0, 2, 3) # Tính tổng trên Batch, H, W
        intersection = torch.sum(probas * targets_1hot, dims)
        cardinality = torch.sum(probas + targets_1hot, dims)
        
        dice = 2. * intersection / (cardinality + 1e-7)
        
        # Lấy mask những index hợp lệ để tính trung bình Dice (bỏ qua ignore_index)
        # Đồng thời bỏ qua lớp Background (103) để model không "ăn gian" bằng cách đoán nền
        valid_class_mask = torch.ones(num_classes, dtype=torch.bool, device=dice.device)
        valid_class_mask[self.background_id] = False
        
        dice_loss = 1.0 - dice[valid_class_mask].mean()
        
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss