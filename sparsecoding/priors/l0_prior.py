import torch

from .prior import Prior


class L0Prior(Prior):
    """Prior with a distribution over the l0-norm of the weights.

    A class of priors where the weights are binary;
    the distribution is over the l0-norm of the weight vector
    (how many weights are active).

    Parameters
    ----------
    prob_distr : Tensor, shape [D], dtype float32
        Probability distribution over the l0-norm of the weights.
    """

    def __init__(
        self,
        prob_distr: torch.Tensor,
    ):
        if prob_distr.dim() != 1:
            raise ValueError(f"`prob_distr` shape must be (D,), got {prob_distr.shape}.")
        if prob_distr.dtype != torch.float32:
            raise ValueError(f"`prob_distr` dtype must be torch.float32, got {prob_distr.dtype}.")
        if not torch.allclose(torch.sum(prob_distr), torch.ones_like(prob_distr)):
            raise ValueError(f"`torch.sum(prob_distr)` must be 1., got {torch.sum(prob_distr)}.")

        self.prob_distr = prob_distr

    @property
    def D(self):
        return self.prob_distr.shape[0]

    def sample(
        self,
        num_samples: int
    ):
        N = num_samples

        num_active_weights = 1 + torch.multinomial(
            input=self.prob_distr,
            num_samples=num_samples,
            replacement=True,
        )  # [N]

        d_idxs = torch.arange(self.D)
        active_idx_mask = (
            d_idxs.reshape(1, self.D)
            < num_active_weights.reshape(N, 1)
        )  # [N, self.D]

        n_idxs = torch.arange(N).reshape(N, 1).expand(N, self.D)  # [N, D]
        # Need to shuffle here so that it's not always the first weights that are active.
        shuffled_d_idxs = [torch.randperm(self.D) for _ in range(N)]
        shuffled_d_idxs = torch.stack(shuffled_d_idxs, dim=0)  # [N, D]

        # [num_active_weights], [num_active_weights]
        active_weight_idxs = n_idxs[active_idx_mask], shuffled_d_idxs[active_idx_mask]

        weights = torch.zeros((N, self.D), dtype=torch.float32)
        weights[active_weight_idxs] += 1.

        return weights
