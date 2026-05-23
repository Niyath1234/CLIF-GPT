"""Neural ODE latent dynamics core: ds/dt = f_psi(s, a)."""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torchdiffeq import odeint, odeint_adjoint

    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False


class JEPAPredictor(nn.Module):
    """Discrete one-step residual predictor (legacy / ablation backend)."""

    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        return_trajectory: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        delta = self.net(torch.cat([state, action], dim=-1))
        predicted = state + delta
        if not return_trajectory:
            return predicted
        trajectory = torch.stack([state, predicted], dim=1)
        return predicted, trajectory


class LatentODEFunc(nn.Module):
    """Vector field f_psi(s, a) for continuous latent state evolution."""

    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, state_dim),
        )
        self._action: torch.Tensor | None = None

    def set_action(self, action: torch.Tensor) -> None:
        self._action = action

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if self._action is None:
            raise RuntimeError("LatentODEFunc.set_action must be called before integration.")
        return self.net(torch.cat([state, self._action], dim=-1))


class NeuralODEJEPA(nn.Module):
    """Integrate ds/dt = f_psi(s, a) over a fixed horizon."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        horizon: float = 1.0,
        num_steps: int = 4,
        method: str = "rk4",
        use_adjoint: bool = True,
    ) -> None:
        super().__init__()
        self.func = LatentODEFunc(state_dim=state_dim, hidden_dim=hidden_dim)
        self.horizon = float(horizon)
        self.num_steps = max(2, int(num_steps))
        self.method = method
        self.use_adjoint = use_adjoint and HAS_TORCHDIFFEQ

    def _time_grid(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.linspace(0.0, self.horizon, self.num_steps, device=device, dtype=dtype)

    def _euler_integrate(self, state: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        trajectory = [state]
        current = state
        dt_steps = times[1:] - times[:-1]
        for dt in dt_steps:
            derivative = self.func(times[0], current)
            current = current + dt * derivative
            trajectory.append(current)
        return torch.stack(trajectory, dim=1)

    def _vector_field(self, action: torch.Tensor):
        """Closure so `action` stays in the autograd graph (module buffers break adjoint w.r.t. a)."""

        def field(_t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
            return self.func.net(torch.cat([state, action], dim=-1))

        return field

    def integrate(self, state: torch.Tensor, action: torch.Tensor, *, use_adjoint: bool | None = None) -> torch.Tensor:
        """Return trajectory tensor [batch, num_steps, state_dim]."""
        times = self._time_grid(state.device, state.dtype)
        adjoint = self.use_adjoint if use_adjoint is None else use_adjoint
        if HAS_TORCHDIFFEQ:
            if adjoint:
                self.func.set_action(action)
                trajectory = odeint_adjoint(self.func, state, times, method=self.method)
            else:
                field = self._vector_field(action)
                trajectory = odeint(field, state, times, method=self.method)
            if trajectory.dim() == 3 and trajectory.shape[0] == self.num_steps:
                trajectory = trajectory.transpose(0, 1)
            return trajectory
        self.func.set_action(action)
        return self._euler_integrate(state, times)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        return_trajectory: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        trajectory = self.integrate(state, action)
        predicted = trajectory[:, -1]
        if not return_trajectory:
            return predicted
        return predicted, trajectory


def create_dynamics(
    dynamics_type: str,
    state_dim: int,
    hidden_dim: int,
    *,
    horizon: float = 1.0,
    num_steps: int = 4,
    method: str = "rk4",
    use_adjoint: bool = True,
) -> nn.Module:
    if dynamics_type == "ode":
        return NeuralODEJEPA(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            horizon=horizon,
            num_steps=num_steps,
            method=method,
            use_adjoint=use_adjoint,
        )
    if dynamics_type in ("mlp", "jepa"):
        return JEPAPredictor(state_dim=state_dim, hidden_dim=hidden_dim)
    raise ValueError(f"Unknown dynamics_type: {dynamics_type}")
