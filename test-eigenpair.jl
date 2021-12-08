using LinearAlgebra
using HomotopyContinuation
include("CIFEN.jl")

m = 1
n = 5
k = 5
@var x[1:n] z

Ts = randn(m+2,n,n)
rs = randn(m,n)
ss = randn(m,n)
v = randn(n)
z₀ = 1.0
V = randn(n,k)
function ϕ(t)
    exp(2*pi*im*t)
end
function ϕprime(t)
    2*pi*im*ϕ(t)
end
h = 0.001
nodes = [h*i for i = 0:1/h]
T = Ts[1,:,:] + z*Ts[2,:,:] + sum([dot(rs[i,:],x)/dot(ss[i,:],x)*Ts[2+i,:,:] for i = 1:m])
highestMoment = 3

startsols = CIFEN.getStartsols(Ts, rs, ss, x, z₀, v)
sys = System(T*x - v, parameters = [z])
sys(startsols[2],[z₀])

function Tfunction(xx,zz)
    sysT = System(T[:])
    reshape(sysT([xx;zz]),size(T))
end
momentMatrices = CIFEN.getMomentMatrices(Ts, rs, ss, x, z, nodes, ϕ, ϕprime, V, highestMoment)
λs,xs = CIFEN.eigenpairsFromIntegrals(Tfunction, 1e-10, momentMatrices...)
println(λs)
println(xs)

T
xs[1]
Tfunction(xs[:,2],λs[2])*xs[:,2]
