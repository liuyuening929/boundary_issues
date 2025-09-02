import torch

class WeightedLoss(torch.nn.Module):
    def __init__(self, method='mse'):
        super(WeightedLoss, self).__init__()
        self.method = method

    def _calc_loss(self, pred, target, weight):

        if self.method == 'mse':

            err = (pred - target) ** 2
            scale = weight * err # this is a weighted error - will mask px that are 0 and apply a weight everywhere else

            if len(torch.nonzero(scale)) != 0:
                pixels_that_are_not_zero = torch.masked_select(scale, torch.gt(weight, 0)) # just calculating the mean from the px that are non-zero
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
