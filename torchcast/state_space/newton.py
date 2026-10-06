"""
Newton refinement of a fitted model's parameters: see :func:`StateSpaceModel.newton_refine()`.
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional, Protocol
from warnings import warn

import torch

from torchcast.internals.hessian import mvnorm_from_hessian


class NewtonObjective(Protocol):
    """
    What :func:`newton_refine` needs from the objective: the (chunked) loss and its derivatives w.r.t. a flat vector of
    parameters. Implemented by ``torchcast.state_space.state_space._ChunkedObjective``.
    """
    param_names: List[str]

    def get_vector(self) -> torch.Tensor:
        ...

    def set_vector(self, vector: torch.Tensor):
        ...

    def loss(self) -> float:
        ...

    def loss_and_grad(self) -> tuple[float, torch.Tensor]:
        ...

    def loss_grad_hessian(self) -> tuple[float, torch.Tensor, torch.Tensor]:
        ...


@dataclass
class NewtonResult:
    """
    The result of :func:`StateSpaceModel.newton_refine()`.

    :param converged: Whether the gradient and step tolerances, or the decrement tolerance, were met.
    :param stop_reason: 'tolerance' (gradient and step), 'decrement', 'max_steps', or 'line_search' (failed to
     decrease the loss).
    :param loss: The final loss (same scale as in :func:`StateSpaceModel.fit()`, i.e. a mean).
    :param params: The final (raw/unconstrained) parameter values, as a flat vector.
    :param param_names: Names for the elements of ``params``, e.g. ``'measure_covariance.cholesky_log_diag[0]'``.
    :param grad: The gradient of the loss at ``params``.
    :param hessian: The hessian of the loss at ``params`` (always computed at the final ``params``, even with
     ``reuse_hessian``).
    :param loss_scale: The loss is a mean; this is the number of elements it's a mean over, so that
     ``hessian * loss_scale`` is the hessian of the summed loss (used for the Laplace approximation). ``None`` for a
     custom ``get_loss``, where this isn't known.
    :param history: One dict per Newton step, with the loss, largest absolute gradient, etc. before the step; and the
     line-search scale and largest absolute (scaled) step taken.
    """
    converged: bool
    stop_reason: str
    loss: float
    params: torch.Tensor
    param_names: List[str]
    grad: torch.Tensor
    hessian: torch.Tensor
    loss_scale: Optional[float]
    history: List[dict] = field(default_factory=list)

    def eigen(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :return: Eigenvalues (ascending) and eigenvectors (columns) of the hessian.
        """
        return torch.linalg.eigh(_symmetrize(self.hessian.double()))

    def weak_directions(self, num: int = 3, num_params: int = 3) -> List[tuple[float, List[tuple[str, float]]]]:
        """
        The directions in parameter-space that the loss is flattest along (smallest eigenvalues of the hessian), which
        flags weakly identified parameters (ridges).

        :param num: The number of directions (eigenvectors) to return.
        :param num_params: For each direction, the number of parameters to return, with the largest absolute loadings.
        :return: A list of ``(eigenvalue, [(param_name, loading), ...])``.
        """
        evals, evecs = self.eigen()
        out = []
        for i in range(min(num, len(evals))):
            vec = evecs[:, i]
            top = vec.abs().argsort(descending=True)[:num_params]
            out.append((evals[i].item(), [(self.param_names[j], vec[j].item()) for j in top.tolist()]))
        return out

    def summed_hessian(self) -> torch.Tensor:
        """
        :return: The hessian of the *summed* loss (rather than the mean), as used for a Laplace approximation.
        """
        if self.loss_scale is None:
            raise RuntimeError(
                "The scale of the loss isn't known for a custom ``get_loss``; use ``get_laplace_mvnorm()`` instead."
            )
        return self.hessian * self.loss_scale

    def laplace_mvnorm(self) -> torch.distributions.MultivariateNormal:
        """
        The Laplace approximation, like :func:`StateSpaceModel.get_laplace_mvnorm()` but reusing the final hessian
        rather than recomputing it.
        """
        return mvnorm_from_hessian(self.params, self.summed_hessian())


def saddle_free_step(grad: torch.Tensor,
                     evals: torch.Tensor,
                     evecs: torch.Tensor,
                     eig_floor: float,
                     max_step: float = float('inf')) -> tuple[torch.Tensor, float]:
    """
    Saddle-free Newton step: ``-V diag(1 / |lambda|) V^T g``, with ``|lambda|`` floored at ``eig_floor * max|lambda|``.
    Using the absolute eigenvalues makes this a descent direction even when the hessian isn't positive definite. The
    step is scaled so that no parameter changes by more than ``max_step``.

    :return: The step, and the Newton decrement ``g^T |H|^-1 g / 2``: the decrease in the loss predicted by the
     (uncapped) step.
    """
    abs_evals = evals.abs()
    floor = max(eig_floor * abs_evals.max().item(), torch.finfo(evals.dtype).tiny)
    abs_evals = abs_evals.clamp(min=floor)
    coefs = -(evecs.T @ grad.to(evecs.dtype)) / abs_evals
    decrement = .5 * (coefs ** 2 * abs_evals).sum().item()
    step = evecs @ coefs
    max_abs_step = step.abs().max().item()
    if max_abs_step > max_step:
        step = step * (max_step / max_abs_step)
    return step, decrement


def newton_refine(objective: NewtonObjective,
                  max_steps: int = 10,
                  grad_tol: float = 1e-5,
                  step_tol: float = 1e-4,
                  max_step: float = 1.,
                  eig_floor: float = 1e-6,
                  decrement_tol: Optional[float] = 1e-3,
                  reuse_hessian: int = 0,
                  loss_scale: Optional[float] = None,
                  verbose: bool = True) -> NewtonResult:
    """
    See :func:`StateSpaceModel.newton_refine()`.
    """
    history = []
    hess = evals = evecs = None
    hess_age = 0  # number of steps since `hess` was computed
    converged = False
    stop_reason = 'max_steps'
    for i in range(max_steps + 1):
        if hess is None or hess_age >= reuse_hessian:
            loss, grad, hess = objective.loss_grad_hessian()
            evals, evecs = _eigh(hess)
            hess_age = 0
        else:
            loss, grad = objective.loss_and_grad()
            hess_age += 1
        if not math.isfinite(loss) or not torch.isfinite(grad).all():
            raise RuntimeError(f"Non-finite loss/gradient at the start of newton step {i}.")

        step, decrement = saddle_free_step(grad, evals, evecs, eig_floor, max_step)
        decrement *= (loss_scale or 1.)
        max_abs_step = step.abs().max().item()
        max_abs_grad = grad.abs().max().item()
        if max_abs_grad < grad_tol and max_abs_step < step_tol:
            converged, stop_reason = True, 'tolerance'
        elif decrement_tol is not None and decrement < decrement_tol:
            converged, stop_reason = True, 'decrement'
        if converged or i == max_steps:
            break

        record = {
            'step': i,
            'loss': loss,
            'max_abs_grad': max_abs_grad,
            'decrement': decrement,
            'eig_min': evals[0].item(),
            'eig_max': evals[-1].item(),
            'num_negative': int((evals < 0).sum()),
            'hessian_age': hess_age,
        }
        scale = _line_search(objective, loss, grad, step)
        record.update(ls_scale=scale, max_abs_step=max_abs_step * (scale or 0.))
        history.append(record)
        if verbose:
            print(_format_record(record))
        if scale is None:
            if hess_age:
                hess = None  # retry with a fresh hessian
                continue
            warn(f"Newton line-search failed to decrease the loss at step {i}; stopping.")
            stop_reason = 'line_search'
            break

    if hess_age:
        # final hessian should be at the final params (for diagnostics, laplace):
        loss, grad, hess = objective.loss_grad_hessian()
        evals, evecs = _eigh(hess)

    result = NewtonResult(
        converged=converged,
        stop_reason=stop_reason,
        loss=loss,
        params=objective.get_vector().detach().clone(),
        param_names=list(objective.param_names),
        grad=grad,
        hessian=hess,
        loss_scale=loss_scale,
        history=history,
    )
    if verbose:
        print(
            f"Newton: {'converged' if converged else 'did not converge'} ({stop_reason}) after {len(history)} steps; "
            f"loss {loss:.6g}, max|grad| {grad.abs().max().item():.3g}. Weakest directions (smallest eigenvalues):"
        )
        for ev, loadings in result.weak_directions():
            print(f"  {ev:.3g}: " + ", ".join(f"{nm} ({ld:+.2f})" for nm, ld in loadings))
    return result


def _line_search(objective: NewtonObjective,
                 loss: float,
                 grad: torch.Tensor,
                 step: torch.Tensor,
                 c1: float = 1e-4,
                 max_halvings: int = 30) -> Optional[float]:
    """
    Backtracking (halving) line-search on the loss alone, accepting on sufficient decrease. A ``LinAlgError`` or
    non-finite loss counts as a rejection. Leaves the objective's params at the accepted point, or restores them.

    :return: The accepted scale for ``step``, or None if none was accepted.
    """
    start = objective.get_vector().detach().clone()
    slope = (grad.double() @ step.double()).item()  # negative, for a descent direction
    # near the optimum, the expected decrease can be smaller than float-error in the loss; so allow for that:
    noise = 10 * torch.finfo(start.dtype).eps * max(1., abs(loss))
    scale = 1.
    for _ in range(max_halvings):
        objective.set_vector(start + scale * step.to(start.dtype))
        try:
            new_loss = objective.loss()
        except torch.linalg.LinAlgError:
            new_loss = float('inf')
        if math.isfinite(new_loss) and new_loss <= loss + c1 * scale * slope + noise:
            return scale
        scale /= 2
    objective.set_vector(start)
    return None


def _eigh(hess: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.linalg.eigh(_symmetrize(hess.double()))


def _symmetrize(mat: torch.Tensor) -> torch.Tensor:
    return (mat + mat.T) / 2


def _format_record(record: dict) -> str:
    reused = f" (reused hessian, age {record['hessian_age']})" if record['hessian_age'] else ""
    return (
        f"Newton step {record['step']}: loss {record['loss']:.6g}; max|grad| {record['max_abs_grad']:.3g}; "
        f"decrement {record['decrement']:.3g}; "
        f"eigenvalues [{record['eig_min']:.3g}, {record['eig_max']:.3g}] ({record['num_negative']} negative){reused}; "
        f"line-search scale {record['ls_scale']}; max|step| {record['max_abs_step']:.3g}"
    )
