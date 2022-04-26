using LinearAlgebra
using HomotopyContinuation
using Random
using Plots; # pgfplotsx();
include("PEPv.jl")

# Experiment 1a
Random.seed!(1)
n=2
@var x[1:2] z

T = [x[1]^2*x[2] -im*x[1]^2*x[2]*2*cos(z);
-x[2]^2*cos(z^2) x[2]^2*3*sin(3*z)];

function ϕ(t)
    a=1.103*π
    b=1.0
    center = 0
    center + a*cos(2*pi*t) + im*b*sin(2*pi*t)
end

function ϕprime(t)
    a = 1.103*π
    b = 1.0
    2*pi*a*sin(2*pi*t) - 2*pi*im*b*cos(2*pi*t)
end

plotnodes = LinRange(0,1,100);
plot(real(ϕ.(plotnodes)), imag(ϕ.(plotnodes)),label="Contour")


function Tfunction(xx,zz)
    sysT = System(T[:])
    reshape(sysT([xx;zz]),size(T))
end

V = PEPv.getRandomMonomialsFor(T,x,z,n);
nodes = LinRange(0,1,100);
highestMoment = 21;
momentMatrices = PEPv.getMomentMatrices(T, x, z, nodes, ϕ, ϕprime, V, highestMoment)
λs,xs,res = PEPv.eigenpairsFromIntegrals(Tfunction, 1e-1, momentMatrices...); 
scatter!(real(λs), imag(λs),label="Beyn", markersize=3,marker=:o)


