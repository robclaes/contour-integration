using LinearAlgebra
using HomotopyContinuation
using Random
using Plots; # pgfplotsx();
include("PEPv.jl")

Random.seed!(1)
n=10;
m=2;
@var x[1:n] z

r = randn(ComplexF64,m,n);
s = randn(ComplexF64,m,n);
A = randn(ComplexF64,n,n);
B = randn(ComplexF64,n,n);
TT = randn(ComplexF64,m,n,n)
rx = r*x;
sx = s*x;
T = A+z*B + sum([rx[i]/sx[i]*TT[i,:,:] for i = 1:m])
model = PEPv.RationalModel(A,B,TT,r,s)


function ϕ(t)
    a=1.0
    b=1.0
    center = 1.0-1.0im
    center + a*cos(2*pi*t) + im*b*sin(2*pi*t)
end
function ϕprime(t)
    a = 1.0
    b = 1.0
    2*pi*a*sin(2*pi*t) - 2*pi*im*b*cos(2*pi*t)
end

function Tfunction(xx,zz)
    sysT = System(T[:])
    reshape(sysT([xx;zz]),size(T))
end

nodes = LinRange(0,1,400);
highestMoment = 35;
V = randn(ComplexF64,n,n);
momentMatrices = PEPv.getMomentMatrices(T, model, x, z, nodes, ϕ, ϕprime, V, highestMoment)
λs,xs,res = PEPv.eigenpairsFromIntegrals(Tfunction, 1e-0, momentMatrices...); λs

function computeEigenvalues(A, B, TT, r, s, x)
    @var zz[1:m] λ
    Sys1 = (A + λ*B+ sum([zz[i]*TT[i,:,:] for i = 1:m]))*x
    rx  = r*x;
    sx = s*x;
    Sys2 = [sx[i]*zz[i]-rx[i] for i = 1:m]
    sys = System(subs([Sys1;Sys2],x[1]=>1))
    R = solve(sys)
    eigenvals = [sol[end] for sol ∈ solutions(R)]
    eigenvec = [[1;sol[1:n-1]] for sol ∈solutions(R)]
    return eigenvals, eigenvec
end
