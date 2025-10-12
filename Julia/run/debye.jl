# --- Setup ---
# First, include the modules from your other files.
include("../src/simulation.jl")
# Note: PdfSampler.jl is assumed to be included within simulation.jl

# Bring the modules into the current scope
import .PICSim
using ProfileView

function main()
    # --- Physical and Simulation Parameters ---
    n_cells = 64
    n_particles_per_cell = 200

    # Plasma parameters
    n0 = 1e23  # Background plasma density [m^-3]
    Te = 1e8   # Electron temperature [K]

    electron_plasma_frequency = sqrt(n0 * Constants.q_electron^2 / (Constants.epsilon0 * Constants.m_electron))
    inv_w_pe = 1 / electron_plasma_frequency
    debye_length = sqrt((Constants.epsilon0 * Constants.k_b * Te) / (n0 * Constants.q_electron^2))
    v_T = sqrt(Constants.k_b * Te / Constants.m_electron) # Electron thermal velocity

    # Domain and weighting parameters
    L = 20 * debye_length       # Simulation domain length
    wq = 0.1 * L
    xc = L / 2
    weight_factor = (L / n_cells)^3 * n0 / n_particles_per_cell

    # Define background charge and initial distribution using anonymous functions.
    # Note the broadcasting dots `.` which apply the operations element-wise to vectors.
    background_charge_density = x -> 0.5*Constants.q_electron*n0 .* (1 .+ L .* exp.(-(x .- xc).^2 ./ (2*wq^2)) ./ (wq*sqrt(2*pi)))

    f0(x, v) = exp.(-v.^2 ./ (2*v_T^2)) ./ (sqrt(2*pi) * v_T * L)

    # Time and particle shape parameters
    dt = 0.005 * inv_w_pe
    end_time = dt * 1e4
    shape = Filters.SplineFilter(3)

    # --- Simulation Instantiation ---
    println("✅ Initializing simulation...")
    sim = PICSim.Simulation(
        n_nodes = n_cells,
        n_particles_per_cell = n_particles_per_cell,
        particle_weight_factor = weight_factor,
        particle_shape_function = shape,
        dt = dt,
        end_time = end_time,
        f0 = f0,
        x_domain = (0.0, L),
        v_domain = (-10 * v_T, 10 * v_T),
        background_charge_density = background_charge_density,
        snapshot_interval = 10
    )

    # --- Run and Post-process ---
    filename = "debye_shielding_$(n_cells)c$(n_particles_per_cell)ppc"

    PICSim.run!(sim)
    PICSim.save_snapshots_to_csv(sim, filename)

    # You can uncomment the line below to show the potential animation instead
    # PICSim.show_potential(sim, fps=15, n0=n0, debye_length=debye_length, save_animation=false)

    println("🎬 Generating and saving animation...")
    PICSim.show_snapshots(sim, fps=15, save_animation=true, filename=filename, show_moments=false)

    println("🎉 Script finished.")
end

main()