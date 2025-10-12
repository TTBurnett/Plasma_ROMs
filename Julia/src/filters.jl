module Filters

using BSplines
using FFTW
using SpecialFunctions
using LinearAlgebra

abstract type Filter end



"""
    SplineFilter

A callable struct that wraps a B-spline basis function from BSplines.jl,
allowing it to be scaled, shifted, and evaluated like a regular function.
"""
struct SplineFilter{Func} <: Filter
    spline_func::Func
    width::Float64
    offset::Float64
    coefficient::Float64
end

function SplineFilter(order::Int; width::Float64=1.0, offset::Float64=0.0, coefficient::Float64=1.0)
    knots = (-0.5*order:0.5*order)
    basis = BSplines.BSplineBasis(order, knots)
    b_spline = basis[order]
    spline_func(x) = coefficient * b_spline((x - offset) / width)
    return SplineFilter(spline_func, width, offset, coefficient)
end

(sf::SplineFilter)(x::Union{Real, AbstractArray}) = sf.spline_func.(x)


# -----------------------------------------------------------------------------
# SiacFilter Definition (no changes here)
# -----------------------------------------------------------------------------

"""
    SiacFilter

A Super Convergent, Alias-Corrected (SIAC) filter. ⚛️
"""
struct SiacFilter <: Filter
    splines::Vector{SplineFilter}
end

function SiacFilter(; n_moments::Int, b_spline_order::Int, width::Float64=1.0)
    @assert iseven(n_moments) "Number of moments must be an even number."
    
    RS = n_moments ÷ 2
    numspline = n_moments + 1
    A = zeros(numspline, numspline)
    
    for m_py in 0:(numspline-1)
        for gamma_py in 0:(numspline-1)
            component = 0.0
            for n_py in 0:m_py
                jsum = sum(
                    (-1)^(j_py + b_spline_order - 1) * binomial(b_spline_order - 1, j_py) *
                    ((j_py - 0.5*(b_spline_order - 2))^(b_spline_order + n_py) - (j_py - 0.5*b_spline_order)^(b_spline_order + n_py))
                    for j_py in 0:(b_spline_order-1)
                )
                component += binomial(m_py, n_py) * (gamma_py - RS)^(m_py - n_py) * factorial(n_py) / factorial(n_py + b_spline_order) * jsum
            end
            A[m_py + 1, gamma_py + 1] = component
        end
    end
    
    b = zeros(numspline)
    b[1] = 1.0
    c = A \ b
    
    splines = [SplineFilter(b_spline_order; width=width, offset=(i - 1 - RS) * width, coefficient=coeff) for (i, coeff) in enumerate(c)]
    
    return SiacFilter(splines)
end

(sf::SiacFilter)(x::Real) = sum(spline(x) for spline in sf.splines)
(sf::SiacFilter)(x_vec::AbstractArray) = sum(spline(x_vec) for spline in sf.splines)

export SplineFilter, SiacFilter, FFTW

end #End of Filters module
    