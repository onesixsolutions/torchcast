from typing import Sequence, Optional

import torch


def hessian(output: torch.Tensor,
            inputs: Sequence[torch.Tensor],
            allow_unused: bool = False,
            create_graph: bool = False,
            progress: bool = False) -> torch.Tensor:
    """
    Adapted from https://github.com/mariogeiger/hessian
    """

    assert output.ndimension() == 0

    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)

    n = sum(input.numel() for input in inputs)
    out = output.new_zeros(n, n)

    if progress is True:
        from tqdm import tqdm
        progress = tqdm
    if progress:
        progress = progress(total=n)

    ai = 0
    for i, inp in enumerate(inputs):
        if not inp.requires_grad:
            continue
        [grad] = torch.autograd.grad(output, inp, create_graph=True, allow_unused=allow_unused)
        grad = torch.zeros_like(inp) if grad is None else grad
        grad = grad.contiguous().view(-1)

        for j in range(inp.numel()):
            if grad[j].requires_grad:
                row = _gradient(grad[j], inputs[i:], retain_graph=True, create_graph=create_graph)[j:]
            else:
                row = grad[j].new_zeros(sum(x.numel() for x in inputs[i:]) - j)

            out[ai, ai:].add_(row.type_as(out))  # ai's row
            if ai + 1 < n:
                out[ai + 1:, ai].add_(row[1:].type_as(out))  # ai's column
            del row
            ai += 1
            if progress:
                progress.update()
        del grad

    return out


def _gradient(outputs: Sequence[torch.Tensor],
              inputs: Sequence[torch.Tensor],
              grad_outputs: Sequence[Optional[torch.Tensor]] = None,
              retain_graph: Optional[bool] = None,
              create_graph: bool = False) -> torch.Tensor:
    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)

    grads = torch.autograd.grad(outputs, inputs, grad_outputs,
                                allow_unused=True,
                                retain_graph=retain_graph,
                                create_graph=create_graph)

    grads = [x if x is not None else torch.zeros_like(y) for x, y in zip(grads, inputs)]

    return torch.cat([x.contiguous().view(-1) for x in grads])