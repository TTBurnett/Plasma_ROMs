include("../src/pdfsampler.jl")

# Bring the necessary modules into scope
using .PdfSampler
using GLMakie

# --- Define some 2D PDFs to sample from ---
gaussian(x, y) = exp(-0.5 * (x^2 + y^2)) / (2 * pi)
tophat(x, y) = 1.0
sin_wave(x, y) = 1 + sin((x + y) * pi)
gaussian_torus(x, y) = exp(-2 * (sqrt(x^2 + y^2) - 3)^2)
pdfs = [gaussian, tophat, sin_wave, gaussian_torus]

for pdf in pdfs
    # --- Sample points from the PDF ---
    println("Sampling from the PDF...")
    samples = PdfSampler.sample(pdf; ranges=[(-5, 5), (-5, 5)], n_samples=20000)
    x_samples = samples[1, :]
    y_samples = samples[2, :]
    println("Sampling complete.")

    # --- Plot the results using GLMakie ---

    # Create the scatter plot
    fig_scatter = Figure(size = (800, 800))
    ax_scatter = Axis(fig_scatter[1, 1],
        title = "Sampled Points from 2D Sine Wave PDF",
        xlabel = "X",
        ylabel = "Y",
        aspect = DataAspect() # This makes the aspect ratio equal
    )
    GLMakie.scatter!(ax_scatter, x_samples, y_samples,
        markersize = 2,
        strokewidth = 0,
        color = (:blue, 0.5) # Set color and alpha transparency
    )
    GLMakie.xlims!(ax_scatter, -5, 5)
    GLMakie.ylims!(ax_scatter, -5, 5)

    # Display the scatter plot in an interactive window
    println("Displaying scatter plot...")
    display(GLMakie.Screen(), fig_scatter)
    # save("pdf_samples.png", fig_scatter) # You can optionally save it

    # Create the histogram
    fig_hist = Figure()
    ax_hist = Axis(fig_hist[1, 1],
        title = "Histogram of X-axis Samples",
        xlabel = "X value",
        ylabel = "Frequency"
    )
    hist!(ax_hist, x_samples, bins = 50)

    # Display the histogram in a new interactive window
    println("Displaying histogram...")
    display(GLMakie.Screen(), fig_hist)
    # save("pdf_histogram.png", fig_hist) # You can optionally save it
end