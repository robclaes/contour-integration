using LinearAlgebra
using HomotopyContinuation

include("PEPv.jl")

@var x[1:3] z

β = [-310 959 774 1389 1313; -365 755 917 1451 1269; -413 837 838 1655 1352]

T = [ 0 (β[1,3]+β[1,5]*z^2)*x[2]+β[1,4]*z*x[3] (β[1,1]+β[1,2]*z^2)*x[3];
       (β[2,3]*x[1]+β[2,4]*x[2])*x[3]^2  (β[2,2]*x[3]^2 + β[2,5]*x[1]^2)*x[2] β[2,1]*x[3]^3;
      (β[3,2]+β[3,5]*z^2)*x[1]+β[3,4]*z*x[3]  0  (β[3,1] + β[3,3]*z^2)*x[3]]

nodes = LinRange(0,1,100);
highestMoment = 5;

function ϕ(t)
    a=0.4
    b=0.4
    center = 0
    center + a*cos(2*pi*t) + im*b*sin(2*pi*t)
end
function ϕprime(t)
    a = 0.4
    b = 0.4
    2*pi*a*sin(2*pi*t) - 2*pi*im*b*cos(2*pi*t)
end

V = PEPv.getRandomMonomialsFor(T,x,z,3);
momentMatrices = PEPv.getMomentMatrices(T, x, z, nodes, ϕ, ϕprime, V, highestMoment);

function Tfunction(xx,zz)
    sysT = System(T[:])
    reshape(sysT([xx;zz]),size(T))
end

λs,xs,res = PEPv.eigenpairsFromIntegrals(Tfunction, 1e-0, momentMatrices...); λs

xs[:,1] = xs[:,1]/xs[3,1];
xs[:,2] = xs[:,2]/xs[3,2];
xs[:,3] = xs[:,3]/xs[3,3];
xs[:,4] = xs[:,4]/xs[3,4];

realsols = real.([xs[1,:]';λs';xs[2,:]'])
