include("Solver.jl")
using Statistics
using ProgressBars
using Plots
gr()

function junction(t, y, paramlist)
    ħ, e, Idc, C, R, I₁ = paramlist
    ϕ, dϕ = y
    v = (ħ/(2*e)) * dϕ
    ddϕ = Idc/C - (1/R)*(v/C) - (I₁/C) * sin(ϕ)
    return [dϕ, ddϕ]
end

αlist = 0.1:0.01:0.12

tin = 0.0
tfin = 1.0
h = 1e-4
funcs = junction

R = 1
C = 1
ħ = 1.0545718176461565e-34
e = 1.602176634e-19
Kb = 1.380649e-23
T = 10
ϕ₀, dϕ₀ = varlist = [0.0 0.0;]
figure = plot()

for γ in 1:2:10
    local vlist = []
    local I₁ = (e*Kb*T)*(γ/ħ)
    println("Averaging time $tin to $tfin for γ = $γ:")
    for α in ProgressBar(αlist)
        local Idc = α * I₁
        local paramlist = [ħ, e, Idc, C, R, I₁]
        global varlist = runge(tin, tfin, h, funcs, varlist, paramlist)

        local ϕ = varlist[:, 1]
        local dϕ = varlist[:, 2]
        plot!(ϕ)
        local avgnum = round(Int, 0.1 * length(varlist[:, 1]))
        local avgvec = last(dϕ, avgnum)
        local vavg = mean(skipmissing((ħ/(2*e)).*(dϕ)))
        push!(vlist, vavg)
    end
    local η = vlist ./ (I₁*R)
    # plot!(η, αlist, label="γ = $γ")
end
