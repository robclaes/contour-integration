using LinearAlgebra
using HomotopyContinuation
using Random
using Plots; pgfplotsx();

include("PEPv.jl")

@var x[1:3] z
T = [x[1]+z*x[2] z*x[2]+x[3] x[1]-x[3];
     x[1]+(1+z)*x[2] (1-z^2)*x[2]-z*x[3] x[1]+x[3];
     (1+z)*x[1]+x[2] x[2]-x[3] z*x[1]+(1-z)*x[3]];

R = solve( System(subs(T*x, x[1]=>1)) );


## Conic plot
# ll = [real(r[end]) for r ∈ solutions(R) if abs(imag(r[end]))<1e-10 && real(r[end])-0.5919<1e-2] 

# conics = subs(T*x,z=>ll[1], x[1]=>1)
# f1(a,b) = evaluate(conics,x[2]=>a, x[3]=>b)[1]
# f2(a,b) = evaluate(conics,x[2]=>a, x[3]=>b)[2]
# f3(a,b) = evaluate(conics,x[2]=>a, x[3]=>b)[3]

# implicit_plot(f1,xlims=(-2.5,0), ylims=(-2.5,0));
# implicit_plot!(f2,xlims=(-2.5,0), ylims=(-2.5,0));
# implicit_plot!(f3,xlims=(-2.5,0), ylims=(-2.5,0));
# savefig("conics_ex1_1.tex");

## Contour plot

eigenvalues = [sol[end] for sol ∈ solutions(R)];
scatter(real(eigenvalues), imag(eigenvalues), aspect_ratio=:equal);

function ϕ(t)
     a=0.4
     b=0.3
     center = 0.6
     center + a*cos(2*pi*t) + im*b*sin(2*pi*t)
 end

plotnodes = LinRange(0,1,50);
plot!(real(ϕ.(plotnodes)), imag(ϕ.(plotnodes)),label="Contour");

savefig("example1-1-domain.tex");
