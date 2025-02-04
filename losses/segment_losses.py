import torch
import torch.nn as nn

# sin_seg loss for weighted BCE+Dice loss
# def sin_seg_loss(pred, gt, weight=None, num_classes=2):
#     loss_metrics = {}
#     bce = nn.BCELoss(pred, gt)
#     dice = DiceLoss(num_classes)
    
#     total_loss = weight*bce + dice
#     loss_metrics['bce'] = bce
#     loss_metrics['dice'] = dice
#     loss_metrics['total_loss'] = total_loss
#     return loss_metrics

# Dice loss
# class DiceLoss(nn.Module):
#     def __init__(self, n_classes):
#         super(DiceLoss, self).__init__()
#         self.n_classes = n_classes

#     def _one_hot_encoder(self, input_tensor):
#         tensor_list = []
#         for i in range(self.n_classes):
#             temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
#             tensor_list.append(temp_prob.unsqueeze(1))
#         output_tensor = torch.cat(tensor_list, dim=1)
#         return output_tensor.float()

#     def _dice_loss(self, score, target):
#         target = target.float()
#         smooth = 1e-5
#         intersect = torch.sum(score * target)
#         y_sum = torch.sum(target * target)
#         z_sum = torch.sum(score * score)
#         loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
#         loss = 1 - loss
#         return loss

#     def forward(self, inputs, target, weight=None, softmax=False):
#         if softmax:
#             inputs = torch.softmax(inputs, dim=1)
#         # target = self._one_hot_encoder(target)
#         if weight is None:
#             weight = [1] * self.n_classes
#         # assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
#         class_wise_dice = []
#         loss = 0.0
#         for i in range(0, self.n_classes):
#             dice = self._dice_loss(inputs[:, i], target[:, i])
#             class_wise_dice.append(1.0 - dice.item())
#             loss += dice * weight[i]
#         return loss / self.n_classes

class DiceLoss(nn.Module):
    def __init__(self, smooth=1., eps=1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.contiguous()
        target = target.contiguous()
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        loss = (1 - ((2. * intersection + self.smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + self.smooth + self.eps)))
        return loss.mean()

class sin_seg_loss(nn.Module):
    def __init__(self, weight=0.5, num_classes=1):
        super(sin_seg_loss, self).__init__()
        self.weight = weight
        self.num_classes = num_classes
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(self, pred, gt):
        if torch.isnan(pred).any() or torch.isnan(gt).any() or torch.isinf(pred).any() or torch.isinf(gt).any():
            print("Pred contains NaN: ", torch.isnan(pred).any())
            print("Target contains NaN: ", torch.isnan(gt).any())
            print("Pred contains Inf: ", torch.isinf(pred).any())
            print("Target contains Inf: ", torch.isinf(gt).any())
            raise ValueError('pred or gt is nan or inf!')
        loss_metrics = {}
        gt = gt.unsqueeze_(1)
        bce = self.bce(pred, gt)
        dice = self.dice(pred, gt)
        total_loss = self.weight*bce + (1-self.weight)*dice
        loss_metrics['bce'] = bce
        loss_metrics['dice'] = dice
        loss_metrics['total_loss'] = total_loss
        return loss_metrics
    

    def get_dice(self, pred, gt):
        return self.dice(pred, gt)
    
    def get_bce(self, pred, gt):
        return self.bce(pred, gt)