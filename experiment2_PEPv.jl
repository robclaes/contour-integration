using LinearAlgebra
using HomotopyContinuation
using Random
using Plots; # pgfplotsx();
include("PEPv.jl")

# Experiment 1a
Random.seed!(1)
n=10
@var x[1:n] z
E,C = exponents_coefficients((1+z)^5*sum(x)^1,[z;x])
mons12 = [prod([z;x].^E[:,i]) for i = 1:size(E,2)]
T = [dot(randn(length(mons12)),mons12) for i = 1:n^2]
T = reshape(T,n,n)

function ϕ(t)
    a=20
    b=1
    center = -10im
    center + a*cos(2*pi*t) + im*b*sin(2*pi*t)
end

function ϕprime(t)
    a = 20
    b = 1
    2*pi*a*sin(2*pi*t) - 2*pi*im*b*cos(2*pi*t)
end

R = solve( System(subs(T*x, x[1]=>1)), threading = false , compile=false);
eigenvalues = [r[end] for r ∈ solutions(R) if abs(r[1]*r[2])>1e-6];
scatter(real(eigenvalues), imag(eigenvalues),label="Exact", markersize=3)#,xlims=(-1,2),ylims=(-0.5,0.5),aspect_ratio=:equal)
plotnodes = LinRange(0,1,100);
plot!(real(ϕ.(plotnodes)), imag(ϕ.(plotnodes)),label="Contour")


function Tfunction(xx,zz)
    sysT = System(T[:])
    reshape(sysT([xx;zz]),size(T))
end

nodes = LinRange(0,1,100);
highestMoment = 19;
V = PEPv.getRandomMonomialsFor(T,x,z,n);
momentMatrices = PEPv.getMomentMatrices(T, x, z, nodes, ϕ, ϕprime, V, highestMoment)
λs,xs,res = PEPv.eigenpairsFromIntegrals(Tfunction, 1e-0, momentMatrices...); λs

scatter!(real(λs), imag(λs),label="Beyn", markersize=3,marker=:x)







# Timing test functions

using BenchmarkTools
function bruteforce_solve()
    d = 2
    e = 13
    n = 3
    Random.seed!(1)
    @var x[1:n] z
    E,C = exponents_coefficients((1+z)^e*sum(x)^d,[z;x])
    mons12 = [prod([z;x].^E[:,i]) for i = 1:size(E,2)]
    T = [dot(randn(length(mons12)),mons12) for i = 1:n^2]
    T = reshape(T,n,n)

    function ϕ(t)
        a=0.25
        b=0.25
        center = -0.2
        center + a*cos(2*pi*t) + im*b*sin(2*pi*t)
    end

    function ϕprime(t)
        a = 0.25
        b = 0.25
        2*pi*a*sin(2*pi*t) - 2*pi*im*b*cos(2*pi*t)
    end
    R = solve(System(subs(T*x, x[1]=>1)));
end

function contour_solve()
    d = 2
    e = 13
    n = 3
    Random.seed!(1)
    @var x[1:n] z
    E,C = exponents_coefficients((1+z)^e*sum(x)^d,[z;x])
    mons12 = [prod([z;x].^E[:,i]) for i = 1:size(E,2)]
    T = [dot(randn(length(mons12)),mons12) for i = 1:n^2]
    T = reshape(T,n,n)

    function ϕ(t)
        a=0.6
        b=0.25
        center = 0.55
        center + a*cos(2*pi*t) + im*b*sin(2*pi*t)
    end

    function ϕprime(t)
        a = 0.6
        b = 0.25
        2*pi*a*sin(2*pi*t) - 2*pi*im*b*cos(2*pi*t)
    end

    function Tfunction(xx,zz)
        sysT = System(T[:])
        reshape(sysT([xx;zz]),size(T))
    end
    
    nodes = LinRange(0,1,50);
    highestMoment = 11;
    V = PEPv.getRandomMonomialsFor(T,x,z,n);
    momentMatrices = PEPv.getMomentMatrices(T, x, z, nodes, ϕ, ϕprime, V, highestMoment)
    λs,xs,res = PEPv.eigenpairsFromIntegrals(Tfunction, 1e-0, momentMatrices...);
end
@btime bruteforce_solve();
@btime contour_solve();