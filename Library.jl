# Gauss Jordan
function gauss_jordan(coeff::Matrix, cons::Vector)
    rows = 1:size(coeff)[1]
    for row in rows
        for other in deleteat!(collect(rows), row)
            mult = coeff[other, row] / coeff[row, row]
            coeff[other, :] -= coeff[row, :] * mult
            cons[other, :] -= cons[row, :] * mult
        end
    end
    norm = sum!(collect(rows), coeff)
    return cons ./ norm
end

# LU decomposition
using LinearAlgebra
function lu_decom(A::Matrix)
    L = Matrix{Float64}(I, size(A))
    U = A
    for i in 1:size(A)[1]
        for j in i+1:size(A)[2]
            L[j, i] = U[j, i] / U[i, i]
            U[j, :] -= L[j, i] .* U[i, :]
        end
    end
    return L, U
end

# Gauss Siedel
function gauss_siedel(A, b, x)
    for i in 1:5
        oldx = x
        for i in 1:length(b)
            summ = [sum(A[i, 1:i-1] .* x[1:i-1]); sum(A[i, i+1:end] .* oldx[i+1:end])]
            x[i] = (1/A[i, i]) * (b[i] - sum(summ))
        end
    end
    return x
end

