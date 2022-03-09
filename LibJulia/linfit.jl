using CSV
using DataFrames

data = CSV.read("vol.csv", DataFrame)
x = data[!, "time(ns)"]
y = data[!, "voltage(volts)"]
z = data[!, "uncertainty(volts)"]
y = log.(y)
include("Library.jl")
using .Statistics

yfit, m, c, q, w, e, r = Main.Statistics.linear_regression(x, y, z, true)

using Plots
scatter(x, y)
plot!(x, yfit)

Main.Statistics.chisq(y, yfit, z)