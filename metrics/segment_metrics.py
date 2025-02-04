import torch
import torch.nn as nn
import numpy as np

from medpy import metric

class SegmentationMetrics(nn.Module):
    def __init__(self, ):
        super(SegmentationMetrics, self).__init__()
        
    def forward(self, pred_mask, true_mask):
        # pred_mask = F.sigmoid(pred_mask)
        pred_mask = torch.sigmoid(pred_mask)
        pred_mask= pred_mask.detach().cpu().numpy()
        pred_mask[pred_mask > 0.5] = 1
        pred_mask[pred_mask <= 0.5] = 0
        self.pred_mask = pred_mask
        true_mask = true_mask.cpu().numpy()
        true_mask[true_mask > 0.5] = 1
        true_mask[true_mask <= 0.5] = 0
        self.true_mask = true_mask

    def f1_score(self):
        """
        Calculate F1 Score, also known as Dice Coefficient.
        """
        # intersection_sum = torch.sum(self.intersection).float()
        # f1_score = (2 * intersection_sum) / (torch.sum(self.true_mask).float() + torch.sum(self.pred_mask).float()) if intersection_sum != 0 else 0
        dice_score = metric.binary.dc(self.pred_mask, self.true_mask)
        return dice_score
    
    def hausdorff_distance(self):
        """
        Calculate Hausdorff Distance.
        """
        hausdorff_distance = metric.binary.hd(self.pred_mask, self.true_mask)
        return hausdorff_distance
    
    def hausdorff_95(self):
        """
        Calculate 95th percentile of Hausdorff Distance.
        """
        if 0 == np.count_nonzero(self.pred_mask):
            print("pred_mask is empty")
            return 0
        if 0 == np.count_nonzero(self.true_mask):
            print("true_mask is empty")
            return 0
        hausdorff_95 = metric.binary.hd95(self.pred_mask, self.true_mask)
        return hausdorff_95
    
    def asd(self):
        """
        Calculate Average Surface Distance.
        """
        if 0 == np.count_nonzero(self.pred_mask):
            print("pred_mask is empty")
            return 0
        if 0 == np.count_nonzero(self.true_mask):
            print("true_mask is empty")
            return 0
        asd = metric.binary.asd(self.pred_mask, self.true_mask)
        return asd
    
    def iou_score(self):
        """
        Calculate Intersection over Union (IoU) score.
        """
        # intersection_sum = torch.sum(self.intersection).float()
        # union_sum = torch.sum(self.union).float()
        # iou_score = intersection_sum / union_sum if union_sum != 0 else 0
        
        # Ensure the masks are boolean
        true_mask = self.true_mask.astype(np.bool_)
        pred_mask = self.pred_mask.astype(np.bool_)

        # Calculate intersection and union
        intersection = np.logical_and(true_mask, pred_mask)
        union = np.logical_or(true_mask, pred_mask)

        # Calculate IoU
        iou_score = np.sum(intersection) / np.sum(union)

        return iou_score.item()