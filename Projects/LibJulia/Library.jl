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
    n = size(A)[1]
    x = zeros(n)
    if prop=="forward"
        for i in 1:n
            x[i] = (b[i] - A[i, 1:i-1]' * x[1:i-1]) / A[i, i]
        end
    else
        for i in reverse(1:n)
            x[i] = (b[i] - A[i, i+1:n]' * x[i+1:n]) / A[i, i]
        end
    end
    return x
end

"""
LU decomposition
"""
function lu_decom(A::AbstractMatrix{Float64}, b::Vector{Float64}, returnlu=false)
    nn = size(A)
    L = Matrix{Float64}(LinearAlgebra.I, nn)
    U = copy(A)
    for i in 1:nn[1]
        for j in i+1:nn[2]
            L[j, i] = U[j, i] / U[i, i]
            U[j, :] -= L[j, i] .* U[i, :]
        end
    end
    y = sub_helper(L, b)
    x = sub_helper(U, y, "backward")
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
using LinearAlgebra
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

function givens(A, k, l)
    β = (A[l, l] - A[k, k]) / (2*A[k, l])
    t = sign(β) / (abs(β) + sqrt(β^2 + 1))
    c = 1/sqrt(t^2+1)
    s = c * t
    return c, s
end

"""
Jacobi Method (with Givens rotation)
"""
function jacobi(A::AbstractMatrix)
    n = size(A)[1]
    Id = Matrix{Float64}(I, n, n)
    
    for l in 1:n
        for k in l+1:n
            if A[l, k] == 0
                println(k,l)
                continue
            end
            s, c = givens(A, k, l)
            S = copy(Id)
            S[k, l] = s
            S[l, k] = -s
            S[k, k] = c
            S[l, l] = c
            A = S'*A*S
        end
    end
    return A
end
end

module Statistics
export linear_regression, chisq
"""
Linear Regression
"""
function linear_regression(x::Vector{Float64}, y::Vector{Float64}, yerr::Vector{Float64}, errfull=false)
    S =  sum(1 ./ (yerr.^2))
    Sx = sum(x ./ (yerr.^2))
    Sy = sum(y ./ (yerr.^2))
    Sxx = sum((x.^2) ./ (yerr.^2))
    Syy = sum((y.^2) ./ (yerr.^2))
    Sxy = sum((x.*y) ./ (yerr.^2))
    
    Δ = S*Sxx - (Sx)^2
    c = (Sxx*Sy - Sx*Sxy)/Δ
    m = (S*Sxy - Sx*Sy)/Δ
    yfit = c .+ m.*x

    σ²c = Sxx/Δ
    σ²m = S/Δ
    cov = -Sx/Δ
    r² = Sxy/(Sxx*Syy)

    if errfull
        return yfit, m, c, σ²m, σ²c, cov, r²
    else
        return yfit, m, c
    end
end

"""
Linear Chi Squared
"""
function chisq(yexp::Vector{Float64}, yfit::Vector{Float64}, yerr::Vector{Float64})
    χsq = sum(((yexp - yfit).^2) ./ yerr)
    χsqn = χsq / (size(yexp)[1]-2)
    return χsq, χsqn
end
end

module Fourier
export discrete
"""
Discrete Fourier Transform
"""
function discrete(x::Vector{ComplexF64})
    n = size(x)[1]
    X = zeros(ComplexF64, n,)
    for k in 0:n-1
        for i in 0:n-1
            X[k+1] += x[i+1] * exp(-(2π*im*i*k)/n)
        end
    end
    return X
end
end
