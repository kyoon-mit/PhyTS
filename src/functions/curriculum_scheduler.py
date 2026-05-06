"""Schedules a scalar loss weight (lambda) over training epochs.

Verbatim port of ``LambdaScheduler`` from the original neutrino_project
``denoising`` branch
([anonymous repository, available upon acceptance]).
"""

import numpy as np


class LambdaScheduler:
    """Schedules a scalar loss weight (lambda) over training epochs.

    Supported schedule types
    ------------------------
    ``'constant'``
        The weight is fixed at ``start_value`` for all epochs.

    ``'linear'``
        Linearly interpolates from ``start_value`` (epoch 0) to
        ``end_value`` (epoch ``total_epochs``).

    ``'cosine'``
        Cosine annealing from ``start_value`` down (or up) to
        ``end_value`` over ``total_epochs``.

    ``'step'``
        Uses a list of ``[epoch_threshold, value]`` breakpoints supplied via
        ``step_schedule``.  The returned value is the one whose threshold is
        the largest value <= ``current_epoch``.  Example::

            step_schedule = [[0, 1.0], [30, 0.5], [60, 0.1]]

        This gives lambda = 1.0 for epochs 0-29, 0.5 for 30-59, 0.1 from 60
        onwards.

    Parameters
    ----------
    schedule_type : str
        One of ``'constant'``, ``'linear'``, ``'cosine'``, ``'step'``.
    start_value : float
        Starting lambda (also the fixed value for ``'constant'``).
    end_value : float
        Target lambda at ``total_epochs`` (used by ``'linear'`` and
        ``'cosine'``; ignored by ``'constant'`` and ``'step'``).
    total_epochs : int
        Total number of training epochs.
    step_schedule : list of [int, float], optional
        Required for ``'step'`` schedule.  Each entry is
        ``[epoch_threshold, lambda_value]``.  Must include an entry at
        epoch 0 (or earlier) to define the initial value.
    """

    def __init__(self,
                 schedule_type: str = 'constant',
                 start_value: float = 1.0,
                 end_value: float = 1.0,
                 total_epochs: int = 100,
                 step_schedule=None):
        self.schedule_type = schedule_type.lower()
        self.start_value   = start_value
        self.end_value     = end_value
        self.total_epochs  = total_epochs

        if self.total_epochs <= 0:
            raise ValueError("total_epochs must be a positive integer.")

        if self.schedule_type == 'step':
            if not step_schedule:
                raise ValueError("'step' schedule requires a non-empty step_schedule list.")
            self._steps = sorted((int(e), float(v)) for e, v in step_schedule)
        else:
            self._steps = None

    def get_lambda(self, current_epoch: int) -> float:
        """Return the scheduled lambda value for *current_epoch*."""
        if self.schedule_type == 'constant':
            return float(self.start_value)

        if self.schedule_type == 'step':
            value = self._steps[0][1]
            for threshold, v in self._steps:
                if current_epoch >= threshold:
                    value = v
            return value

        progress = min(current_epoch / self.total_epochs, 1.0)

        if self.schedule_type == 'linear':
            return float(self.start_value + (self.end_value - self.start_value) * progress)

        if self.schedule_type == 'cosine':
            factor = 0.5 * (1.0 - np.cos(np.pi * progress))
            return float(self.start_value + (self.end_value - self.start_value) * factor)

        raise ValueError(f"Unknown schedule type: '{self.schedule_type}'. "
                         "Choose 'constant', 'linear', 'cosine', or 'step'.")
