import torch
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM

im1 = torch.rand([1, 1, 256, 256], generator=torch.manual_seed(42))
im2 = im1 * 0.75

metric = SSIM(return_full_image=True, reduction=None)
score = metric(im1, im2)
print(score)
