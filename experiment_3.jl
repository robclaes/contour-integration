using LinearAlgebra
using HomotopyContinuation
using Random
using Statistics
using Plots;  #pgfplotsx();
include("PEPv.jl")

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


function Tfunction(xx,zz)
    sysT = System(T[:])
    reshape(sysT([xx;zz]),size(T))
end

V = PEPv.getRandomMonomialsFor(T,x,z,n);
nodes = LinRange(0,1,150);
highestMoment = 15;
momentMatrices = PEPv.getMomentMatrices(T, x, z, nodes, ϕ, ϕprime, V, highestMoment)
λs,xs,res = PEPv.eigenpairsFromIntegrals(Tfunction, 1e-1, momentMatrices...); 
scatter!(real(λs), imag(λs),label="Beyn", markersize=3,marker=:o);
savefig("experiment3.tex");



nodelengths = [50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,230,240];
maxres = zeros(Float64,size(nodelengths))
minres = zeros(Float64,size(nodelengths))
medres = zeros(Float64,size(nodelengths))
nbsols= zeros(Float64,size(nodelengths))
i=1;
for nodelength ∈ nodelengths
    println(nodelength)
    nodes = LinRange(0,1,nodelength);
    highestMoment = 15;
    momentMatrices = PEPv.getMomentMatrices(T, x, z, nodes, ϕ, ϕprime, V, highestMoment)
    λs,xs,res = PEPv.eigenpairsFromIntegrals(Tfunction, 1e-1, momentMatrices...); 
    maxres[i] = maximum(res);
    minres[i] = minimum(res);
    medres[i] = median(res);
    nbsols[i] = length(res);
    i +=1;
end

plot(nodelengths,minres, yaxis=:log,yticks=[1e-5, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15] , label="Min");
plot!(nodelengths,medres, yaxis=:log,yticks=[1e-5, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15] , label="Med");
plot!(nodelengths,maxres, yaxis=:log,yticks=[1e-5, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15] , label="Max");
savefig("experiment3_conv.tex")