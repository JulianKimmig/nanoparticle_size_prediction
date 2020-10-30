import warnings

import torch
from torch.nn import _reduction as _Reduction
from torch.nn.modules.loss import _Loss


def rell1_loss(input, target, size_average=None, reduce=None, reduction="mean"):
    # type: (Tensor, Tensor, Optional[bool], Optional[bool], str) -> Tensor
    r"""l1_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor

    Function that takes the mean element-wise absolute value difference.

    See :class:`~torch.nn.L1Loss` for details.
    """
    if not (target.size() == input.size()):
        warnings.warn(
            "Using a target size ({}) that is different to the input size ({}). "
            "This will likely lead to incorrect results due to broadcasting. "
            "Please ensure they have the same size.".format(
                target.size(), input.size()
            ),
            stacklevel=2,
        )
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    target[target == 0] = 10 ** -32
    ret = torch.abs((input - target) / target)
    if reduction != "none":
        ret = torch.mean(ret) if reduction == "mean" else torch.sum(ret)
    return ret


class RMAE(_Loss):

    __constants__ = ["reduction"]

    def __init__(self, size_average=None, reduce=None, reduction="mean"):
        super(RMAE, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        loss = rell1_loss(input, target, reduction=self.reduction)
        return loss
