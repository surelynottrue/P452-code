function Kcollapse(y::Vector, h::Float64, kval::Vector)
    multiplier = (1/6) * h * [1, 2, 2, 1]
    return sum((multiplier .* kval)[:, :])
end

function K(t::Float64, h::Float64, vars::Vector, params::Vector, Kiter::Int64, Kvals::Vector)
    (Kiter == 3) && (push!(Kvals, funcs(t+h, vars + h*Kvals[end], params)); return (Kvals))
    (Kiter == 1) && push!(Kvals, funcs(t, vars, params))
    push!(Kvals, funcs(t+h/2, vars + (h/2)*Kvals[end], params))
    K(t, h, vars, params, Kiter+1, Kvals)
end

function runge(tin::Float64, tfin::Float64, h::Float64, funcs::Function, varlist::VecOrMat, params::Vector)
    for time in tin+h:h:tfin
        Kvals = K(time, h, varlist[end, :], params, 1, [])
        yval = varlist[end, :] + Kcollapse(varlist[end, :], h, Kvals)
        varlist = vcat(varlist, transpose(yval))
    end
    return varlist
end