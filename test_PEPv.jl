using LinearAlgebra
using HomotopyContinuation
using Random
include("PEPv.jl")

function ϕ(t)
    1+0.2*exp(2*pi*im*t)
end

function ϕprime(t)
    0.4*pi*im*ϕ(t)
end



# Example 1
@var x[1:3] z

T = [x[1] -x[2] z*x[3];
     x[3] 5*(z^2-z)*x[1] -x[2];
     x[1]-z*x[2] x[3] z*x[1]-2*x[2]    
    ]

function Tfunction(xx,zz)
    sysT = System(T[:])
    reshape(sysT([xx;zz]),size(T))
end



nodes = LinRange(0,1,100);
highestMoment = 3;
V = randn(3,2);
momentMatrices = PEPv.getMomentMatrices(T, x, z, nodes, ϕ, ϕprime, V, highestMoment)
λs,xs = PEPv.eigenpairsFromIntegrals(Tfunction, 1e-2, momentMatrices...)