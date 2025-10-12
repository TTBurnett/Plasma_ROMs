# First, include your custom module files.
# Julia will look for these files in the same directory.
include("constants.jl")
include("pdfsampler.jl")
include("filters.jl")

# -----------------------------------------------------------------------------
# PIC Simulation Module
# -----------------------------------------------------------------------------
module PICSim

# Import necessary packages
using Plots
using Printf
using SparseArrays
using SpecialFunctions
using LinearAlgebra
using DelimitedFiles
using Trapz
using Statistics

# Import the modules we included in the main script.
# The `..` tells Julia to look in the parent scope.
using ..PdfSampler
using ..Constants
import ..Filters

# --- Core Simulation Structs ---

# Struct to hold a single snapshot of the simulation state
struct Snapshot
    time::Float64
    x::Vector{Float64}
    v::Vector{Float64}
    phi::Vector{Float64}
end

# Mutable struct to hold the entire simulation's state and parameters
mutable struct Simulation{SF <: Filters.Filter}
    # Parameters
    n_nodes::Int
    n_particles::Int
    end_time::Float64
    dt::Float64
    x_domain::Tuple{Float64, Float64}
    v_domain::Tuple{Float64, Float64}
    L::Float64
    dx::Float64
    weight_factor::Float64
    shape_function::SF
    snapshot_interval::Int
    
    # State Variables
    time::Float64
    px::Vector{Float64}
    pv::Vector{Float64}
    colors::Vector{String}
    node_positions::Vector{Float64}
    bg_charge_density::Vector{Float64}
    snapshots::Vector{Snapshot}

    # Internal fields (initialized during run)
    shape_function_width::Int
    n_idx_tiling::Vector{Int}
    p_idx::Vector{Int}
    phi_matrix::Matrix{Float64}
    electric_field_matrix::Matrix{Float64}

    # Fields updated every timestep
    interpolation::SparseMatrixCSC{Float64, Int}
    nrho::Vector{Float64}
    ne_field::Vector{Float64}
    pe_field::Vector{Float64}

    # Constructor function
    function Simulation(;
            n_nodes::Int, n_particles_per_cell::Int,
            dt::Float64, end_time::Float64,
            f0::Function, x_domain, v_domain,
            background_charge_density,
            particle_weight_factor::Float64,
            particle_shape_function::SF,
            snapshot_interval::Int=10,
            color_rule=nothing) where {SF <: Filters.Filter}
        
        n_particles = n_particles_per_cell * n_nodes
        L = x_domain[2] - x_domain[1]
        dx = L / n_nodes
        xv = PdfSampler.sample(f0; ranges=[x_domain, v_domain], n_samples=n_particles)
        px = xv[1, :]
        pv = xv[2, :]

        colors = if color_rule !== nothing
            color_rule(px, pv)
        else
            ["cyan" for _ in 1:n_particles]
        end

        node_positions = collect(range(x_domain[1] + 0.5*dx, stop=x_domain[2] - 0.5*dx, length=n_nodes))
        bg_charge_density_vec = background_charge_density(node_positions)

        # Initialize fields with placeholder values
        new{SF}(n_nodes, n_particles, end_time, dt, x_domain, v_domain, L, dx,
            particle_weight_factor, particle_shape_function, snapshot_interval,
            0.0, px, pv, colors, node_positions, bg_charge_density_vec, [],
            0, Int[], Int[], zeros(n_nodes, n_nodes), zeros(n_nodes, n_nodes), # internal fields
            spzeros(n_nodes, n_particles), zeros(n_nodes), zeros(n_nodes), zeros(n_particles) # loop fields
        )
    end
end

# --- Helper Functions ---

# Periodically shift particle positions into the main domain [x_min, x_max)
function shift_x_to_domain(sim::Simulation, x)
    return @. mod((x - sim.x_domain[1]), sim.L) + sim.x_domain[1]
end

# Calculate the shortest periodic distance between a particle and a node
function get_distance(sim::Simulation, px, nx)
    px_domain = shift_x_to_domain(sim, px)
    return @. 0.5 * sim.L - abs(abs(px_domain - nx) - 0.5 * sim.L)
end

# --- Core Physics and Update Functions ---

# Construct the sparse matrix that maps particle quantities to grid nodes
function get_interpolation_matrix(sim::Simulation, x::Vector{Float64})
    # Calculate the nearest node index (1-based) for each particle
    base_n_idx = floor.(Int, ((x .- sim.x_domain[1]) .% sim.L) ./ sim.dx) .+ 1

    # Tile and repeat indices to account for shape function width
    n_idx = repeat(base_n_idx, inner=sim.shape_function_width) .+ sim.n_idx_tiling
    n_idx = mod1.(n_idx, sim.n_nodes) # mod1 for 1-based periodic indexing

    interp_vals = sim.shape_function.(get_distance(sim, x[sim.p_idx], sim.node_positions[n_idx]) ./ sim.dx)
    
    return sparse(n_idx, sim.p_idx, interp_vals, sim.n_nodes, sim.n_particles)
end

# Calculate fluid moments (density, momentum, energy) on the grid
function get_moments(sim::Simulation, px::Vector{Float64}, pv::Vector{Float64})
    px_shifted = shift_x_to_domain(sim, px)
    moments = zeros(3, sim.n_nodes)
    interpolation = get_interpolation_matrix(sim, px_shifted)

    # Note: sum over particles is `dims=2`
    moments[1, :] = sum(interpolation, dims=2)[:] .* (Constants.m_electron * sim.weight_factor / sim.dx)
    moments[2, :] = (interpolation * pv) .* (Constants.m_electron * sim.weight_factor / sim.dx)
    moments[3, :] = (interpolation * (pv .^ 2)) .* (0.5 * Constants.m_electron * sim.weight_factor / sim.dx)
    return moments
end

# The following functions (ending in '!') modify the simulation state `sim`
function push_particles!(sim::Simulation)
    sim.px .+= sim.pv .* sim.dt
end

function interpolate_particles_to_field!(sim::Simulation)
    sim.interpolation = get_interpolation_matrix(sim, sim.px)
    particle_charge_density = sum(sim.interpolation, dims=2)[:]
    sim.nrho = -Constants.q_electron * sim.weight_factor .* particle_charge_density ./ sim.dx^3 .+ sim.bg_charge_density
end

function update_electric_field!(sim::Simulation)
    mul!(sim.ne_field, sim.electric_field_matrix, sim.nrho)
end

function interpolate_field_to_particles!(sim::Simulation)
    mul!(sim.pe_field, transpose(sim.interpolation), sim.ne_field)
end

function accelerate_particles!(sim::Simulation)
    sim.pv .+= -Constants.q_over_m * sim.dt .* sim.pe_field
end

function update!(sim::Simulation)
    push_particles!(sim)
    interpolate_particles_to_field!(sim)
    update_electric_field!(sim)
    interpolate_field_to_particles!(sim)
    accelerate_particles!(sim)
    sim.time += sim.dt
end

function save_snapshot!(sim::Simulation)
    phi = sim.phi_matrix * sim.nrho
    snapshot = Snapshot(sim.time, copy(sim.px), copy(sim.pv), phi)
    push!(sim.snapshots, snapshot)
end

# --- Main Simulation Runner ---

function run!(sim::Simulation; save_snapshots=true)
    # --- One-time setup ---
    y1 = sim.shape_function.(range(-100, 101))
    width = count(x -> abs(x) > 1e-9, y1)
    
    relative_n_idx = (-(width÷2)):(width÷2)
    sim.shape_function_width = length(relative_n_idx)
    sim.n_idx_tiling = repeat(collect(relative_n_idx), sim.n_particles)
    sim.p_idx = repeat(1:sim.n_particles, inner=sim.shape_function_width)

    # Setup finite-difference operators with periodic boundary conditions
    # A is the discrete Laplacian (d^2/dx^2)
    A = spdiagm(
        -1 => ones(sim.n_nodes - 1),
         0 => -2 * ones(sim.n_nodes),
         1 => ones(sim.n_nodes - 1)
    )
    A[1, sim.n_nodes] = 1
    A[sim.n_nodes, 1] = 1
    
    # B is the discrete gradient (-d/dx)
    B = spdiagm(
        -1 => -1 * ones(sim.n_nodes - 1),
         1 =>  1 * ones(sim.n_nodes - 1)
    )
    B[1, sim.n_nodes] = -1
    B[sim.n_nodes, 1] = 1
    
    # Calculate matrices to get potential (phi) and E-field from charge density (rho)
    sim.phi_matrix = -sim.dx^2 * pinv(Matrix(A)) / Constants.epsilon0
    sim.electric_field_matrix = -(0.5 / sim.dx) * B * sim.phi_matrix
    
    # --- Main Loop ---
    println("Starting simulation...")
    start_time = time()
    while sim.time < sim.end_time
        update!(sim)
        current_step = round(Int, sim.time / sim.dt)
        if current_step % sim.snapshot_interval == 0
            percentage = sim.time / sim.end_time * 100
            @printf("t = %.3g/%.3g (%.2f%%)\n", sim.time, sim.end_time, percentage)
            if save_snapshots
                save_snapshot!(sim)
            end
        end
    end
    elapsed = time() - start_time
    @printf("Simulation finished. Elapsed time: %.4f seconds\n", elapsed)
end

# --- Visualization and Export Functions ---

function show_snapshots(sim::Simulation; fps=10, save_animation=false, filename="PIC_simulation", show_moments=true)
    println("Generating animation...")
    
    moment_array = nothing
    if show_moments
        # Pre-calculate all moments for stable y-axis limits
        all_moments = [get_moments(sim, s.x, s.v) for s in sim.snapshots]
        moment_array = permutedims(cat(all_moments..., dims=3), (1, 2, 3))
    end

    unique_colors = unique(sim.colors)

    anim = @animate for (t_idx, s) in enumerate(sim.snapshots)
        if !show_moments
            p = plot(title=@sprintf("Time = %.3g", s.time),
                     xlabel="x (m)", ylabel="\$v_x\$ (m/s)",
                     xlims=sim.x_domain, ylims=sim.v_domain, legend=false)
            for color in unique_colors
                mask = sim.colors .== color
                if any(mask)
                    scatter!(p, shift_x_to_domain(sim, s.x[mask]), s.v[mask], alpha=0.5, color=color, markersize=3)
                end
            end
            plot(p) # Required for @animate
        else
            p1 = plot(title=@sprintf("Time = %.3g", s.time), ylabel="\$v_x\$ (m/s)", ylims=sim.v_domain, legend=false)
            for color in unique_colors
                mask = sim.colors .== color
                if any(mask)
                     scatter!(p1, shift_x_to_domain(sim, s.x[mask]), s.v[mask], alpha=0.5, color=color, markersize=2)
                end
            end

            moments = moment_array[:, :, t_idx]
            p2 = plot(sim.node_positions, moments[1, :], ylabel="Density", ylims=extrema(moment_array[1,:,:]), legend=false)
            p3 = plot(sim.node_positions, moments[2, :], ylabel="Momentum", ylims=extrema(moment_array[2,:,:]), legend=false)
            p4 = plot(sim.node_positions, moments[3, :], ylabel="Energy", ylims=extrema(moment_array[3,:,:]), xlabel="x (m)", legend=false)
            
            plot(p1, p2, p3, p4, layout=(4, 1), size=(800, 1000), sharex=true, xlims=sim.x_domain)
        end
    end

    if save_animation
        println("Saving animation to $filename...")
        gif(anim, "$(filename).gif", fps=fps)
    end
    println("Done!")
    return anim # Return animation object for display in environments like Pluto/Jupyter
end

function save_snapshots_to_csv(sim::Simulation, filename_base::String)
    println("Saving snapshots to $(filename_base)*.csv...")
    particle_data = hcat([vcat(s.x, s.v) for s in sim.snapshots]...)
    writedlm("$(filename_base)_particles.csv", particle_data, ',')
    writedlm("$(filename_base)_node_positions.csv", sim.node_positions, ',')
    println("Done!")
end

end # --- End of PICSim Module ---