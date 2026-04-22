"""Bohmian Flow.

Reference implementation for
    "Quantum Dynamics via Score Matching on Bohmian Trajectories"
    Lei Wang, Phys. Rev. Lett. (2026).

The quantum time evolution of a nodeless wave function is reduced to a single
real-valued score field s = grad ln rho.  A neural score network is trained
end-to-end by minimising the Fisher divergence between s_theta and the score
of its own implied density rho_theta.  rho_theta is constructed from the
deformation gradient F = dx(t)/dx(0) of the Bohmian flow, which is
co-integrated alongside the particle trajectories by a symplectic leapfrog
scheme.
"""

__version__ = "1.0.0"
