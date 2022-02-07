"""
Author: Spandan Anupam
Date: Feb 2022
As part of the Computational Physics course at NISER
"""

include("Solver.jl")
using Statistics
using ProgressBars
using Plots
using Unitful
using Distributions
using PhysicalConstants.CODATA2018
gr()

# https://stackoverflow.com/questions/53404705/julia-generate-normally-distributed-random-number-with-restricted-range
function noise(I₁)
    μ = 0
    σ = 2 * I₁
    n = rand(Truncated(Normal(μ, σ), -3σ, 3σ), 1)[1]
    return n
end

function junction(t, y, paramlist)
    ħ, e, Idc, C, R, I₁ = paramlist
    ϕ, dϕ = y
    v = (ħ/(2*e)) * dϕ
    ddϕ = Idc/C - (1/R)*(v/C) - (I₁/C) * sin(ϕ) + noise(I₁)/C
    return [dϕ, ddϕ]
end

αlist = 0.1:0.01:1.0

tin = 0.0
tfin = 1.5e-2
h = 1e-4
funcs = junction


R = 1
C = 2e-5
ħ = ustrip(ReducedPlanckConstant)
e = ustrip(ElementaryCharge)
Kb = ustrip(BoltzmannConstant)
T = 100
ϕ₀, dϕ₀ = varlist = [0.0 0.0;]
# figure = plot()

for γ in (1:1000:1000)
    local vlist = []
    local I₁ = γ*1e-9
    println("Averaging time $tin to $tfin for γ = $γ:")
    for α in ProgressBar(αlist)
        local Idc = α * I₁
        local paramlist = [ħ, e, Idc, C, R, I₁]
        global varlist = runge(tin, tfin, h, funcs, varlist, paramlist)

        local ϕ = varlist[:, 1]
        local dϕ = varlist[:, 2]
        local avgnum = round(Int, 0.1 * length(varlist[:, 1]))
        local avgvec = last(dϕ, avgnum)
        local vavg = mean(skipmissing((ħ/(2*e)).*(dϕ)))
        push!(vlist, vavg)
    end
    local η = vlist ./ (I₁*R)
    plot!(η, αlist, label="γ = $γ")
    global varlist = [0.0 0.0;]
end
1