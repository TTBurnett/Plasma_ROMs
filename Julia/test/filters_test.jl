include("../src/filters.jl")
using .Filters

using GLMakie
GLMakie.activate!()

# --- Setup variables ---
n = 1001
min_x, max_x = -5, 5
x = range(min_x, max_x, length=n)
dx = x[2] - x[1]
width = 0.1
orders = [2, 3] # Corresponds to quadratic and cubic splines

# --- Create a Figure object to hold the plots ---
# This is the main window or canvas for our plots.
fig = Figure(size = (1000, 800))

# --- Generate comparison plots ---
for (i, order) in enumerate(orders)
    # Create the basic spline and the SIAC filter
    spline = Filters.SplineFilter(order; width=width)
    siac_filter = Filters.SiacFilter(n_moments=6, b_spline_order=order, width=width)
    y_spline = spline(x)
    y_siac = siac_filter(x)

    # --- Plot in real space ---
    # Create an Axis in the top row (row 1) and i-th column
    ax_real = Axis(fig[1, i],
        title = "Order $order Spline",
        xlabel = "x",
        ylabel = "Value"
    )
    lines!(ax_real, x, y_spline, label="Spline", color=:black)
    lines!(ax_real, x, y_siac, label="SIAC", color=:black, linestyle=:dash)
    GLMakie.xlims!(ax_real, -0.5, 0.5)
    axislegend(ax_real) # Add a legend to this specific axis

    # --- Calculate and plot the Fourier transform ---
    freq = FFTW.rfftfreq(n, 1/dx)
    yf_spline = abs.(FFTW.rfft(y_spline))
    yf_siac = abs.(FFTW.rfft(y_siac))

    # Create an Axis in the bottom row (row 2) and i-th column
    ax_freq = Axis(fig[2, i],
        title = "Frequency Response",
        xlabel = "Frequency",
        ylabel = "Magnitude"
    )
    lines!(ax_freq, freq, yf_spline / maximum(yf_spline), label="Spline", color=:black)
    lines!(ax_freq, -freq, yf_spline / maximum(yf_spline), label="", color=:black)
    lines!(ax_freq, freq, yf_siac / maximum(yf_siac), label="SIAC", color=:black, linestyle=:dash)
    lines!(ax_freq, -freq, yf_siac / maximum(yf_siac), label="", color=:black, linestyle=:dash)
    GLMakie.xlims!(ax_freq, -5/width, 5/width)
end

# --- Display the final figure ---
# This will open the interactive window.
display(fig)

println("Plot window displayed. Close the window to exit the script.")