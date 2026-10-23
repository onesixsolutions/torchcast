import torch


class RegimeTransition(torch.nn.Module):
    """
    Base-class for how regime-probabilities evolve over time. Subclasses implement :func:`initial` (the regime-prior
    at the first timestep) and :func:`forward` (the regime-prior at the next timestep, given the regime-posterior at
    the current timestep).

    Both receive ``base_probs``, the long-run probability of each combo implied by the :class:`MixtureComponent`
    base-rates: a ``(num_combos,)`` tensor (or, if any components have predictors, a ``(num_groups, num_combos)``
    tensor, for the timestep that the regime-prior is for).
    """

    def __init__(self, num_combos: int):
        super().__init__()
        self.num_combos = num_combos

    def initial(self, base_probs: torch.Tensor, num_groups: int) -> torch.Tensor:
        """
        :param base_probs: A ``(num_combos,)`` or ``(num_groups, num_combos)`` tensor.
        :param num_groups: The number of groups.
        :return: A ``(num_groups, num_combos)`` tensor of regime-probabilities.
        """
        raise NotImplementedError

    def forward(self, posterior: torch.Tensor, base_probs: torch.Tensor) -> torch.Tensor:
        """
        :param posterior: A ``(num_groups, num_combos)`` tensor of regime-probabilities at the current timestep.
        :param base_probs: A ``(num_combos,)`` or ``(num_groups, num_combos)`` tensor, for the next timestep.
        :return: A ``(num_groups, num_combos)`` tensor of regime-probabilities at the next timestep.
        """
        raise NotImplementedError


class StaticTransition(RegimeTransition):
    def initial(self, base_probs, num_groups):
        return base_probs.expand(num_groups, -1)

    def forward(self, posterior, base_probs):
        return base_probs.expand(posterior.shape[0], -1)


class StickyTransition(RegimeTransition):
    """
    Each combo ``i`` persists with probability ``stay_i``; otherwise the next combo is drawn from a shared distribution
    ``b``. That is, ``T[i, j] = stay_i * (i == j) + (1 - stay_i) * b_j``.

    Rather than learning ``b`` directly, it's derived so that the stationary distribution of ``T`` is ``base_probs``
    (``b ∝ base_probs * (1 - stay)``). So ``base_probs`` keeps its interpretation as the long-run probability of each
    combo, and is also the initial distribution. With ``stay = 0`` this reduces to a static mixture where the
    regime-prior is always ``base_probs``. (If the base-rates vary over time -- components with predictors -- then
    ``base_probs`` is the stationary distribution for the current predictors.)

    :param num_combos: The number of regime-combos.
    :param stay_init: Initial value for the probability of persisting in the current combo.
    """

    def __init__(self, num_combos: int, stay_init: float = 0.1):
        super().__init__(num_combos=num_combos)
        assert 0 < stay_init < 1
        stay_init = torch.as_tensor(stay_init, dtype=torch.get_default_dtype())
        self._stay_logit = torch.nn.Parameter(
            (torch.log(stay_init) - torch.log1p(-stay_init)).expand(num_combos).clone()
        )

    def get_stay_prob(self) -> torch.Tensor:
        """
        :return: A ``(num_combos,)`` tensor: the probability of persisting in each combo, ``stay``.
        """
        return torch.sigmoid(self._stay_logit)

    def _jump_probs(self, base_probs: torch.Tensor) -> torch.Tensor:
        b = base_probs * (1 - self.get_stay_prob())
        return b / b.sum(-1, keepdim=True)

    def initial(self, base_probs: torch.Tensor, num_groups: int) -> torch.Tensor:
        return base_probs.expand(num_groups, -1)

    def forward(self, posterior: torch.Tensor, base_probs: torch.Tensor) -> torch.Tensor:
        stay = self.get_stay_prob()
        # equivalent to `posterior @ self.matrix(base_probs)`, without materializing the matrix:
        leave = (posterior * (1 - stay)).sum(-1, keepdim=True)
        return posterior * stay + leave * self._jump_probs(base_probs)

    def matrix(self, base_probs: torch.Tensor) -> torch.Tensor:
        """
        The transition-matrix, for inspection and testing (:func:`forward` applies it without building it).

        :param base_probs: A ``(num_combos,)`` tensor (so not for components with predictors, whose base-rates vary
         by group).
        :return: The ``(num_combos, num_combos)`` transition-matrix.
        """
        stay = self.get_stay_prob()
        return torch.diag(stay) + (1 - stay).unsqueeze(-1) * self._jump_probs(base_probs).unsqueeze(0)
