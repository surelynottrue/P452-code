include("Library.jl")
using .MatInv
using ProgressBars

# Q1: Performance Benchmark
B = [
    1.0 -1.0 4.0 0.0 2.0 9.0; 
    0.0 5.0 -2.0 7.0 8.0 4.0; 
    1.0 0.0 5.0 7.0 3.0 -2.0; 
    6.0 -1.0 2.0 3.0 0.0 8.0;
    -4.0 2.0 0.0 5.0 -5.0 3.0;
    0.0 7.0 -1.0 5.0 4.0 -2.0
]

b = [
    19.0;
    2.0;
    13.0;
    -7.0;
    -9.0;
    2.0
]

@time gauss_jordan(B, b)
@time lu_decom(B, b)
@time jacobi(B, b)
@time gauss_siedel(B, b)

# Q2: Inversion
A = [
    2.0 -3.0 0.0 0.0 0.0 0.0;
    -1.0 4.0 -1.0 0.0 -1.0 0.0;
    0.0 -1.0 4.0 0.0 0.0 -1.0;
    0.0 0.0 0.0 2.0 -3.0 0.0;
    0.0 -1.0 0.0 -1.0 4.0 -1.0;
    0.0 0.0 -1.0 0.0 -1.0 4.0;
]

a = [
    -5/2;
    2/3;
    3;
    -4/3;
    -1/3;
    5/3;
]

using LinearAlgebra

@time gauss_jordan(A, a)
@time lu_decom(A, a)
@time jacobi(A, a)
@time gauss_siedel(A, a)

function inverse(A, func)
    n = size(A)[1]
    amat = Matrix{Float64}(I, n, n)
    for i in ProgressBar(1:n)
        a = amat[:, i]
        amat[:, i] = func(A,a)
    end
    return amat
end

## Be sure to remove the progressbar before timing
@time inverse(A, gauss_jordan) * A

# Q3: Inversion without storage
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

A = Yoda(100, 100, (i,j)->id(i,j,μ,m))

@time inverse(A, jacobi)