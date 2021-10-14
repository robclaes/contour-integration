include("eigenpair-extraction.jl");
using LinearAlgebra

A0 = randn(5,5);
A1 = randn(5,5);
A2 = randn(5,5);
A3 = randn(5,5);

T(λ,z) = A0;

l,x = CIFEN.eigenpairsFromIntegrals(T,1e2,A0,A1,A2,A3)
println(l);
println(x);

