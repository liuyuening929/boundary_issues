import torch

class WeightedLoss(torch.nn.Module):
    def __init__(self, method='mse'):
        super(WeightedLoss, self).__init__()
        self.method = method

    def _calc_loss(self, pred, target, mask):

        if self.method == 'mse':

            err = (pred - target) ** 2
            scale = mask * err

            if len(torch.nonzero(scale)) != 0:
                pixels_that_are_not_zero = torch.masked_select(scale, torch.gt(mask, 0))
                loss = torch.mean(pixels_that_are_not_zero)
            else:
                loss = torch.mean(scale)

            return loss
        
        elif self.method == 'bce':
            raise NotImplemented("for later")
        
        else:
            raise AssertionError("u gotta pick mse or bce")


    def forward(self, predictions, targets, mask):

        affs_loss = self._calc_loss(predictions, targets, mask)

        return affs_loss
