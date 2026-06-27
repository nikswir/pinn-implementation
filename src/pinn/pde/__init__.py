"""Heat-equation PINN: physics-informed loss and analytic reference."""

from pinn.pde.heat import HeatPINN
from pinn.pde.analytic import u_exact, u_exact_scalar

__all__ = ["u_exact", "HeatPINN", "u_exact_scalar"]
