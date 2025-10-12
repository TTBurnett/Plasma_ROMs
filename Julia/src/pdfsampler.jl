module PdfSampler

using Random # For random number generation and seeding

export sample

function sample(pdf::Function; ranges::Vector{<:Tuple}, n_samples::Int, seed=nothing)
    # Seed the random number generator if a seed is provided
    if seed !== nothing
        Random.seed!(seed)
    end

    n_dims = length(ranges)
    samples = zeros(Float64, n_dims, n_samples)
    idx = 1 # Julia uses 1-based indexing

    n_grid_points = 100
    coords = [range(r[1], r[2], length=n_grid_points) for r in ranges]
    max_value = maximum(pdf(p...) for p in Iterators.product(coords...))

    while idx <= n_samples
        candidate_sample = [rand() * (r[2] - r[1]) + r[1] for r in ranges]
        y_sample = rand() * max_value

        if y_sample < pdf(candidate_sample...)
            samples[:, idx] = candidate_sample
            idx += 1
        end
    end

    return samples
end

end # --- End of PdfSampler Module ---