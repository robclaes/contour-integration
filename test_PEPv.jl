using LinearAlgebra
using HomotopyContinuation
using Random
include("PEPv.jl")

function ϕ(t)
    2*exp(2*pi*im*t)
end

function ϕprime(t)
    4*pi*im*exp(2*pi*im*t)
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



nodes = LinRange(0,1,50);
highestMoment = 7;
V = randn(3,3);
momentMatrices = PEPv.getMomentMatrices(T, x, z, nodes, ϕ, ϕprime, V, highestMoment,x->[x[1]^2;x[1]*x[2];x[1]*x[3]])
λs,xs = PEPv.eigenpairsFromIntegrals(Tfunction, 1e-10, momentMatrices...)
λs


# Example X
@var x[1:3] z

T = [1 (z) 0;
     (z) 1 0;
     0 0 (z)^2+1]

function Tfunction(xx,zz)
    sysT = System(T[:])
    reshape(sysT([zz]),size(T))
end

function ϕ(t)
    1+0.1*exp(2*pi*im*t)
end

function ϕprime(t)
    0.2*pi*im*exp(2*pi*im*t)
end

nodes = LinRange(0,1,1000);
highestMoment = 1;
V = randn(3,3);
momentMatrices = PEPv.getMomentMatrices(T, x, z, nodes, ϕ, ϕprime, V, highestMoment,x->x);
λs,xs = PEPv.eigenpairsFromIntegrals(Tfunction, 1e-2, momentMatrices...)


# Example 7

T = [1 z 1;
     2 1 z;
    x[2] (z+1)*x[3]+x[2] 0];

function Tfunction(xx,zz)
    sysT = System(T[:])
    reshape(sysT([xx[2:3];zz]),size(T))
end

function ϕ(t)
    -1+sqrt(3) +0.1*exp(2*pi*im*t)
end

function ϕprime(t)
    0.2*pi*im*exp(2*pi*im*t)
end

nodes = LinRange(0,1,1000);
highestMoment = 3;
V = randn(3,3);
momentMatrices = PEPv.getMomentMatrices(T, x, z, nodes, ϕ, ϕprime, V, highestMoment,x->x);
λs,xs = PEPv.eigenpairsFromIntegrals(Tfunction, 1e-2, momentMatrices...)
    