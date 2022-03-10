include("Library.jl")
using .Eigen
using LinearAlgebra

δ(i, j) = (i==j) ? 1.0 : 0.0

mutable struct Yoda<:AbstractMatrix{Float64}
    rows::Int
    cols::Int
    func
end

function Base.getindex(A::Yoda, i::Int, j::Int)
    return A.func(i, j)
end

function Base.size(A::Yoda)
    return A.rows, A.cols
end

μ = 1
m = 0.2

id(i,j, μ, m) = 0.5*(δ(i+μ, j) + δ(i-μ, j) + 2*δ(i, j)) + (m^2)*(δ(i, j))

A = Yoda(5, 5, (i,j)->id(i,j,μ,m))

Main.Eigen.jacobi(A)