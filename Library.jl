"""
Author: Spandan Anupam
As part of the Computational Physics course at NISER
"""
module MatInv
export gauss_jordan, lu_decom, jacobi, gauss_siedel, conjugate_gradient
using Test
using LinearAlgebra
"""
Gauss Jordan
"""
function gauss_jordan(M::AbstractMatrix{Float64}, m::Vector{Float64})
    A = copy(M); b = copy(m)
    rows = 1:size(A)[1]
    for row in rows
        for other in deleteat!(collect(rows), row)
            mult = A[other, row] / A[row, row]
            A[other, :] -= A[row, :] * mult
            b[other, :] -= b[row, :] * mult
        end
    end
    
    norm = sum(A, dims=2)
    return vec(b ./ norm)
end

function sub_helper(A, b, prop="forward")
    n = size(A)[1]; x = zeros(n)
    prop=="forward" ? rng=(1:n) : rng=reverse(1:n)
    for i in rng
        if (i == rng[1]) x[i] = b[i] / A[i, i]
        elseif (prop!="forward") x[i] = (b[i] - A[i, i+1:n]' * x[i+1:n]) / A[i, i]
        else x[i] = (b[i] - A[i, n:i-1]' * x[n:i-1]) / A[i, i] end
    end
    return x
end

"""
LU decomposition
"""
function lu_decom(A::AbstractMatrix{Float64}, b::Vector{Float64}, returnlu=false)
    nn = size(A)
    L = Matrix{Float64}(I, nn)
    U = copy(A)
    for i in 1:nn[1]
        for j in i+1:nn[2]
            L[j, i] = U[j, i] / U[i, i]
            U[j, :] -= L[j, i] .* U[i, :]
        end
    end
    y = sub_helper(L, b)
    x = sub_helper(U, y, "back")
    returnlu ? (return x, L, U) : (return x)
end

"""
Jacobi
"""
function jacobi(A::AbstractMatrix{Float64}, b::Vector{Float64}, x=ones(Float64, size(A)[1], ), stop=1e-9)
    oldx = zeros(Float64, size(A)[1], ) 
    while (oldx - x)' * (oldx - x) > stop
        if ((oldx - x)' * (oldx - x)) > 1e2 
            error("Did not converge, make sure coefficient matrix is diagonal heavy")
            return
        end
        oldx = copy!(oldx, x)
        for i in 1:length(b)
            summ = A[i, 1:i-1]' * x[1:i-1] + A[i, i+1:end]' * x[i+1:end]
            x[i] = (1/A[i, i]) * (b[i] - summ)
        end
    end
    return x
end

"""
Gauss Siedel
"""
function gauss_siedel(A::AbstractMatrix{Float64}, b::Vector{Float64}, x=ones(Float64, size(A)[1], ), stop=1e-9)
    oldx = zeros(Float64, size(A)[1], ) 
    while (oldx - x)' * (oldx - x) > stop
        if ((oldx - x)' * (oldx - x)) > 1e2 
            error("Did not converge, make sure coefficient matrix is diagonal heavy")
            return
        end
        oldx = copy!(oldx, x)
        for i in 1:length(b)
            summ = A[i, 1:i-1]' * x[1:i-1] + A[i, i+1:end]' * oldx[i+1:end]
            x[i] = (1/A[i, i]) * (b[i] - summ)
        end
    end
    return x
end

"""
Conjugate Gradient
"""
function conjugate_gradient(A::AbstractMatrix{Float64}, b::Vector{Float64}, x=ones(Float64, size(A)[1], ), stop=1e-9)
    d = copy(b - A * x)
    r = copy(b - A * x)
    con = 0
    for i in 1:size(A)[1]
        Ad = A * d
        modr = (r' * r)
        α = modr / (d' * Ad)
        x += α * d
        β = 1 / modr
        r -= α * (Ad)
        modr = (r' * r)
        modr < stop ? break : 
        β *= modr
        d = r + β * d
        con += 1
    end
    if con == size(A)[1] error("Passed n loops, possibly did not converge") end
    return x
end 
end

module Eigen
export power, jacobi
"""
Power Method
"""
function power(A::Matrix{Float64}, x=ones(Float64, size(A)[1], ), stop=1e-3)
    v = x; vold = zeros(Float64, size(x))
    λold = 0.0; λ = -1.0
    while abs(λ - λold) > stop
        vold = v; v = A * v
        λold = λ; λ = (v' * x)/(vold' * x)
    end
    return λ, v
end

"""
Jacobi Method (with Givens rotation)
"""
function jacobi(A)
end
end