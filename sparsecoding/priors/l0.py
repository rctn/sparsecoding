import torch

from sparsecoding.priors.common import Prior


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
        if not torch.allclose(torch.sum(prob_distr), torch.ones(1, dtype=torch.float32)):
            raise ValueError(f"`torch.sum(prob_distr)` must be 1., got {torch.sum(prob_distr)}.")

        self.prob_distr = prob_distr

    @property
    def D(self):
        return self.prob_distr.shape[0]

    def sample(
        self,
        num_samples: int,
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

    def log_prob(
        self,
        sample: torch.Tensor,
    ):
        super().check_sample_input(sample)

        l0_norm = torch.sum(sample != 0., dim=1).type(torch.long)  # [num_samples]
        log_prob = torch.log(self.prob_distr[l0_norm - 1])
        log_prob[l0_norm == 0] = -torch.inf
        return log_prob

# TODO: Add L0ExpPrior, where the number of active units is distributed exponentially.

class L0IidPrior(Prior):
    """L0-sparse Prior with non-binary weights.
    
    If a weight is active, its value is drawn from an i.i.d. Prior.

    Parameters
    ----------
    prob_distr : Tensor, shape [D], dtype float32
        Probability distribution over the l0-norm of the weights.
    active_weight_prior : Prior
        The distribution for active weights.
        Since weights drawn from this distribution must be i.i.d.,
            we enforce `active_weight_prior.D` to be 1.
    """

    def __init__(
        self,
        prob_distr: torch.Tensor,
        active_weight_prior: Prior,
    ):
        if prob_distr.dim() != 1:
            raise ValueError(f"`prob_distr` shape must be (D,), got {prob_distr.shape}.")
        if prob_distr.dtype != torch.float32:
            raise ValueError(f"`prob_distr` dtype must be torch.float32, got {prob_distr.dtype}.")
        if not torch.allclose(torch.sum(prob_distr), torch.ones(1, dtype=torch.float32)):
            raise ValueError(f"`torch.sum(prob_distr)` must be 1., got {torch.sum(prob_distr)}.")
        if active_weight_prior.D != 1:
            raise ValueError(
                f"`active_weight_prior.D` must be 1 (got {active_weight_prior.D}). "
                f"This enforces that can sample i.i.d. weights."
            )

        self.prob_distr = prob_distr
        self.active_weight_prior = active_weight_prior

    @property
    def D(self):
        return self.prob_distr.shape[0]

    def sample(
        self,
        num_samples: int,
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
        n_active_idxs = int(torch.sum(active_idx_mask).cpu().numpy())
        active_weight_values = self.active_weight_prior.sample(n_active_idxs)
        weights[active_weight_idxs] += active_weight_values.reshape(-1)

        return weights

    def log_prob(
        self,
        sample: torch.Tensor,
    ):
        super().check_sample_input(sample)

        active_weight_mask = (sample != 0.)

        l0_norm = torch.sum(active_weight_mask, dim=1).type(torch.long)  # [num_samples]
        log_prob = torch.log(self.prob_distr[l0_norm - 1])
        log_prob[l0_norm == 0] = -torch.inf

        active_log_prob = (
            self.active_weight_prior.log_prob(sample.reshape(-1, 1)).reshape(sample.shape)
        )  # [num_samples, D]
        active_log_prob[~active_weight_mask] = 0.  # [num_samples, D]
        log_prob += torch.sum(active_log_prob, dim=1)  # [num_samples]

        return log_prob
