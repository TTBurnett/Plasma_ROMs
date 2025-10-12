module Constants

export q_electron, m_electron, epsilon0, c, k_b, q_over_m

# The values of all the fundamental constants needed for a PIC simulation.

const q_electron = 1.60217663e-19    # Elementary charge [C]
const m_electron = 9.1093837e-31     # Electron rest mass [kg]
const epsilon0 = 8.8541878e-12       # Vacuum permittivity [F/m]
const c = 3.0e8                      # Speed of light in vacuum [m/s]
const k_b = 1.380649e-23             # Boltzmann constant [J/K]

# Pre-calculated charge-to-mass ratio for electrons
const q_over_m = q_electron / m_electron # [C/kg]

end