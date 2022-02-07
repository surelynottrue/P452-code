"""
Author: Spandan Anupam
Date: Feb 2022
As part of the Computational Physics course at NISER
"""

function K(t, h, vars, params)
    K₁ = funcs(t, vars, params)
    K₂ = funcs(t + h/2, vars + (h/2)*K₁, params)
    K₃ = funcs(t + h/2, vars + (h/2)*K₂, params)
    K₄ = funcs(t + h, vars + h*K₃, params)
    return sum([1, 2, 2, 1] .* [K₁, K₂, K₃, K₄])
end

function runge(tin, tfin, h, funcs, varlist, params)
    for time in tin+h:h:tfin
        Kvals = K(time, h, varlist[end, :], params)
        yval = varlist[end, :] + (h/6) * Kvals
        varlist = vcat(varlist, transpose(yval))
    end
    return varlist
end