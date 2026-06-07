"""Heat-equation PINN: physics-informed loss and analytic reference."""

from pinn.pde.analytic import u_exact, u_exact_scalar
from pinn.pde.heat import HeatPINN

__all__ = ["HeatPINN", "u_exact", "u_exact_scalar"]
