include("Library.jl")
using .Fourier
x = randn(ComplexF64, 100)

y = discrete(x)

using Plots
plot(y)