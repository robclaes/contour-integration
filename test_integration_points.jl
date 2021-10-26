using LinearAlgebra
using HomotopyContinuation
using Random
using Plots
include("CIFEN.jl")

Random.seed!(0)

m = 1
n = 5
k = 5
@var x[1:n] z

Ts = randn(ComplexF64,m+2,n,n)
rs = randn(m,n)
ss = randn(m,n)
z₀ = 1.0
V = randn(n,k)
function ϕ(t)
    exp(2*pi*im*t)
end
function ϕprime(t)
    2*pi*im*ϕ(t)
end


T = Ts[1,:,:] + z*Ts[2,:,:] + sum([dot(rs[i,:],x)/dot(ss[i,:],x)*Ts[2+i,:,:] for i = 1:m])
highestMoment = 3

function Tfunction(xx,zz)
    sysT = System(T[:])
    reshape(sysT([xx;zz]),size(T))
end

nb_sols = [];
min_res = [];
max_res = [];
for intPoints ∈ [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    nodes = LinRange(0,1,intPoints)
    momentMatrices = CIFEN.getMomentMatrices(Ts, rs, ss, x, z, nodes, ϕ, ϕprime, V, highestMoment,true)
    λs,xs = CIFEN.eigenpairsFromIntegrals(Tfunction, 1e0, momentMatrices...)
    res =[];
    for i = 1:length(λs)
        push!(res,norm(Tfunction(xs[:,i],λs[i])*xs[:,i]))
    end
    push!(nb_sols,length(λs))
    push!(min_res,isempty(res) ? Inf : minimum(res))
    push!(max_res,isempty(res) ? Inf : maximum(res))
end
intPoints = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
plot(intPoints, min_res, m=:circle, label="Min res")
plot!(intPoints, max_res, m=:circle, label="Max res")
yaxis!("Residual", :log10, minorticks=true)

nb_sols = [];
min_res = [];
max_res = [];
for intPoints ∈ [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120].+1
    nodes = LinRange(0,1,intPoints)
    momentMatrices = CIFEN.getMomentMatrices(Ts, rs, ss, x, z, nodes, ϕ, ϕprime, V, highestMoment,false)
    λs,xs = CIFEN.eigenpairsFromIntegrals(Tfunction, 1e0, momentMatrices...)
    res =[];
    for i = 1:length(λs)
        push!(res,norm(Tfunction(xs[:,i],λs[i])*xs[:,i]))
    end
    push!(nb_sols,length(λs))
    push!(min_res,isempty(res) ? Inf : minimum(res))
    push!(max_res,isempty(res) ? Inf : maximum(res))
end
intPoints = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120].+1
plot!(intPoints, min_res, m=:circle, label="Min res S")
plot!(intPoints, max_res, m=:circle, label="Max res S")
