function K(t::Float64, h::Float64, vars::Vector, params::Vector)
    K₁ = funcs(t, vars, params)
    K₂ = funcs(t + h/2, vars + (h/2)*K₁, params)
    K₃ = funcs(t + h/2, vars + (h/2)*K₂, params)
    K₄ = funcs(t + h, vars + h*K₃, params)
    return sum([1, 2, 2, 1] .* [K₁, K₂, K₃, K₄])
end

function runge(tin::Float64, tfin::Float64, h::Float64, funcs::Function, varlist::VecOrMat, params::Vector)
    for time in tin+h:h:tfin
        Kvals = K(time, h, varlist[end, :], params)
        yval = varlist[end, :] + (h/6) * Kvals
        varlist = vcat(varlist, transpose(yval))
    end
    return varlist
end