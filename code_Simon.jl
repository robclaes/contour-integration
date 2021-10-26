using HomotopyContinuation
using LinearAlgebra

m = 1
n = 5
k = 5
@var x[1:n] z

Ts = randn(ComplexF64,m+2,n,n)
rs = randn(m,n)
ss = randn(m,n)
v = randn(ComplexF64,n)
z₀ = 1.0
V = randn(n,k)
function ϕ(t)
    exp(2*pi*im*t)
end
function ϕprime(t)
    2*pi*im*ϕ(t)
end
h = 0.001
nodes = [h*i for i = 0:1/h]
T = Ts[1,:,:] + z*Ts[2,:,:] + sum([dot(rs[i,:],x)/dot(ss[i,:],x)*Ts[2+i,:,:] for i = 1:m])
highestMoment = 3


# Let Ts be given by Ts[1,:,:] = T01, Ts[2,:,:] = T02, Ts[2+i,:,:] = Ti
#     rs          by rs[i,:] = ri
#     ss          by ss[i,:] = si
function getStartsols(Ts, rs, ss, x, z₀, v)
    m = size(rs,1)
    n = size(Ts[1,:,:],1)
    @var zz[1:m] TT0[1:n,1:n] TT[1:m,1:n,1:n] vv[1:n] rr[1:m,1:n] sss[1:m,1:n]
    parvec = [TT0[:];TT[:];vv[:];rr[:];sss[:]]
    Sys1 = (TT0 + sum([zz[i]*TT[i,:,:] for i = 1:m]))*x - vv
    Sys2 = [dot(sss[i,:],x)*zz[i]-dot(rr[i,:],x) for i = 1:m]
    Sys = System([Sys1;Sys2], parameters = parvec)
    target_T0 = Ts[1,:,:]+z₀*Ts[2,:,:]
    target_T = Ts[3:end,:,:]
    target_v = v
    target_r = rs
    target_s = ss
    target_parvec = [target_T0[:];target_T[:];target_v[:];target_r[:];target_s[:]]
    R3 = solve(Sys,target_parameters = target_parvec)
    return [sol[1:n] for sol ∈ solutions(R3)]
end

startsols = getStartsols(Ts, rs, ss, x, z₀, v)
sys = System(T*x - v, parameters = [z])
sys(startsols[2],[z₀])

function ϕ(t)
    exp(2*pi*im*t)
end
function ϕprime(t)
    2*pi*im*ϕ(t)
end

h = 0.001
nodes = [h*i for i = 0:1/h]

function computeTrace(T,x,z,z₀,startsols,v,ϕ,nodes)
    sys = System(T*x - v, parameters = [z])
    R = solve(sys,startsols;start_parameters = [z₀], target_parameters = [[ϕ(nodes[i])] for i = 1:length(nodes)])
    solution_sets = [solutions(RR[1]) for RR ∈ R]
    traces = [sum(solset) for solset ∈ solution_sets]
    return solution_sets, traces
end

function getTraceMatrices(Ts, rs, ss, x, z, ϕ, nodes, V)
    T = Ts[1,:,:] + z*Ts[2,:,:] + sum([dot(rs[i,:],x)/dot(ss[i,:],x)*Ts[2+i,:,:] for i = 1:m])
    z₀ = ϕ(nodes[1])
    k = size(V,2)
    traceMatrices = zeros(ComplexF64,length(nodes),n,k)
    for i = 1:k
        v = V[:,i]
        startsols = getStartsols(Ts, rs, ss, x, z₀, v)
        solution_sets, traces = computeTrace(T,x,z,z₀,startsols,v,ϕ,nodes)
        for j = 1:length(nodes)
            traceMatrices[j,:,i] = traces[j]
        end
    end
    traceMatrices = [traceMatrices[i,:,:] for i = 1:length(nodes)]
    return traceMatrices
end

function momentMatrix(traceMatrices,nodes,ϕ,ϕprime,i)
    N = length(nodes)-1
    AiN = 1/2*(traceMatrices[1]*ϕprime(nodes[1])*ϕ(nodes[1])^i + traceMatrices[end]*ϕprime(nodes[end])*ϕ(nodes[end])^i)
    for j = 2:N
        AiN += traceMatrices[j]*ϕprime(nodes[j])*ϕ(nodes[j])^i
    end
    AiN = AiN/(im*N)
end

function getMomentMatrices(Ts, rs, ss, x, z, nodes, ϕ, ϕprime, V, highestMoment)
    traceMatrices = getTraceMatrices(Ts, rs, ss, x, z, ϕ, nodes, V)
    momentMatrices = [momentMatrix(traceMatrices,nodes,ϕ,ϕprime,i) for i = 0:highestMoment]
end

function computeEigenvalues(Ts, rs, ss, x, z)
    @var zz[1:m] λ
    Sys1 = (Ts[1,:,:] + λ*Ts[2,:,:] + sum([zz[i]*Ts[2+i,:,:] for i = 1:m]))*x
    Sys2 = [dot(ss[i,:],x)*zz[i]-dot(rs[i,:],x) for i = 1:m]
    sys = System(subs([Sys1;Sys2],x[1]=>1))
    R = solve(sys)
    eigenvals = [sol[end] for sol ∈ solutions(R)]
end

function Tfunction(xx,zz)
    sysT = System(T[:])
    reshape(sysT([xx;zz]),size(T))
end

include("CIFEN.jl")
momentMatrices = getMomentMatrices(Ts, rs, ss, x, z, nodes, ϕ, ϕprime, V, highestMoment)
λs,xs = CIFEN.eigenpairsFromIntegrals(Tfunction, 1e-10, momentMatrices...)
