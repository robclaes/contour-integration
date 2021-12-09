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
v = randn(ComplexF64, n)
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

function computeEigenvalues(Ts, rs, ss, x, z)
    @var zz[1:m] λ
    Sys1 = (Ts[1,:,:] + λ*Ts[2,:,:] + sum([zz[i]*Ts[2+i,:,:] for i = 1:m]))*x
    Sys2 = [dot(ss[i,:],x)*zz[i]-dot(rs[i,:],x) for i = 1:m]
    sys = System(subs([Sys1;Sys2],x[1]=>1))
    R = solve(sys)
    eigenvals = [sol[end] for sol ∈ solutions(R)]
end

eigenvalues = computeEigenvalues(Ts,rs,ss,x,z);

F = System(T*x-v, parameters=[z]);

for target_eigenvalue ∈ eigenvalues
    pert = randn(ComplexF64);
    pert = pert / abs(pert) * 1e-6;
    target_value = target_eigenvalue + pert;
    start_value = target_eigenvalue+randn(ComplexF64);
    startsols = CIFEN.getStartsols(Ts, rs, ss, x, start_value, v);
    R = solve(F,startsols, start_parameters = [start_value], target_parameters = [target_value]);
    sols = solutions(R);
    @assert length(sols) == 5;
    println("Target λ=", target_eigenvalue, ", solutions going to ∞=" ,count(norm.(sols).>1e3));
end
