module CIFEN #Contour Integration For Eigenvector Nonlinearities

using LinearAlgebra
using BlockArrays
using HomotopyContinuation

# Let Ts be given by Ts[1,:,:] = T01, Ts[2,:,:] = T02, Ts[2+i,:,:] = Ti
#     rs          by rs[i,:] = ri
#     ss          by ss[i,:] = si
function getStartsols(Ts, rs, ss, x, z₀, v)
    m = size(rs,1)
    n = size(Ts[1,:,:],1)
    @var zz[1:m] TT0[1:n,1:n] TT[1:m,1:n,1:n] vv[1:n] rr[1:m,1:n] sss[1:m,1:n]
    rrx = rr*x
    sssx = sss*x
    parvec = [TT0[:];TT[:];vv[:];rr[:];sss[:]]
    Sys1 = (TT0 + sum([zz[i]*TT[i,:,:] for i = 1:m]))*x - vv
    Sys2 = [sssx[i]*zz[i] - rrx[i] for i =1:m]
    Syss = System([Sys1;Sys2], parameters = parvec)
    target_T0 = Ts[1,:,:]+z₀*Ts[2,:,:]
    target_T = Ts[3:end,:,:]
    target_v = v
    target_r = rs
    target_s = ss
    target_parvec = [target_T0[:];target_T[:];target_v[:];target_r[:];target_s[:]]
    R3 = solve(Syss,target_parameters = target_parvec)
    return [sol[1:n] for sol ∈ solutions(R3)]
end

function computeTrace(T,x,z,z₀,startsols,v,ϕ,nodes)
    sys = System(T*x - v, parameters = [z])
    R = solve(sys,startsols;start_parameters = [z₀], target_parameters = [[ϕ(nodes[i])] for i = 1:length(nodes)])
    solution_sets = [solutions(RR[1]) for RR ∈ R]
    traces = [sum(solset) for solset ∈ solution_sets]
    return solution_sets, traces
end

function computeTrace_2(T,x,z,z₀,startsols,v,ϕ,nodes)
    sys = System(T*x - v, parameters = [z])
    tracker = Tracker(ParameterHomotopy(sys, [2.2], [2.2]))
    solution_sets = []
    for i = 1:length(nodes)
        start_parameters!(tracker, [z₀])
        target_parameters!(tracker, [ϕ(nodes[i])])
        z₀ = ϕ(nodes[i])
        res = track.(tracker, startsols, 1.0, 0.0)
        startsols = [r.solution for r ∈ res]
        solutions_sets = push!(solution_sets,startsols)
        #println("iteration $i")
    end
    traces = [sum(solset) for solset ∈ solution_sets]
    return solution_sets, traces
end

function getTraceMatrices(Ts, rs, ss, x, z, ϕ, nodes, V)
    m,n=size(rs)
    rsx = rs*x
    ssx = ss*x
    T = Ts[1,:,:] + z*Ts[2,:,:] + sum( [rsx[i]/ssx[i]*Ts[2+i,:,:] for i = 1:m] )
    z₀ = ϕ(nodes[1])
    k = size(V,2)
    traceMatrices = zeros(ComplexF64,length(nodes),n,k)
    for i = 1:k
        v = V[:,i]
        startsols = getStartsols(Ts, rs, ss, x, z₀, v)
        solution_sets, traces = computeTrace_2(T,x,z,z₀,startsols,v,ϕ,nodes)
        for j = 1:length(nodes)
            traceMatrices[j,:,i] = traces[j]
        end
    end
    traceMatrices = [traceMatrices[i,:,:] for i = 1:length(nodes)]
    return traceMatrices
end

function momentMatrix(traceMatrices,nodes,ϕ,ϕprime,i, trap=true)
    N = length(nodes)-1
    if trap
        AiN = 1/2*(traceMatrices[1]*ϕprime(nodes[1])*ϕ(nodes[1])^i + traceMatrices[end]*ϕprime(nodes[end])*ϕ(nodes[end])^i)
        for j = 2:N
            AiN += traceMatrices[j]*ϕprime(nodes[j])*ϕ(nodes[j])^i
        end
        AiN = AiN/(im*N)
        return AiN
    else # Simpson's rule
        AiN = 1/3*(traceMatrices[1]*ϕprime(nodes[1])*ϕ(nodes[1])^i + traceMatrices[end]*ϕprime(nodes[end])*ϕ(nodes[end])^i)
        coeff = [4,2]
        for j = 2:N
            AiN += coeff[j%2+1]/3*traceMatrices[j]*ϕprime(nodes[j])*ϕ(nodes[j])^i
        end
        AiN = AiN/(im*N)
        return AiN
    end
end

function getMomentMatrices(Ts, rs, ss, x, z, nodes, ϕ, ϕprime, V, highestMoment,trap=true)
    traceMatrices = getTraceMatrices(Ts, rs, ss, x, z, ϕ, nodes, V)
    momentMatrices = [momentMatrix(traceMatrices,nodes,ϕ,ϕprime,i,trap) for i = 0:highestMoment]
end

function blockHankel(Ai::Matrix{N} ...) where {N<:Number}
    @assert length(Ai)%2 == 0;

    K::Int = length(Ai)/2;
    n,m = size(Ai[1]);

    if K == 1
        return Ai[1], Ai[2];
    end

    H0 = BlockArray{N}(undef_blocks,n*ones(Int,K),m*ones(Int,K));
    H1 = BlockArray{N}(undef_blocks,n*ones(Int,K),m*ones(Int,K));
    for i = 1:K
        for j = 1:K
            H0[Block(i,j)] = Ai[i+j-1];
            H1[Block(i,j)] = Ai[i+j];
        end
    end
    return Array(H0), Array(H1);
end

function eigenpairsFromIntegrals(T::Function, tol::Real, Ai::Matrix{N} ...) where {N<:Number}
    n = size(Ai[1],1);
    H0,H1 = blockHankel(Ai...);
    K = size(H0,2);
    V,S,W = svd(H0);
    # TODO: more robust rank test
    firstind = findfirst(S.<S[1]*length(S)*eps())
    k = isnothing(firstind) ? K : firstind - 1;
    V0 = V[:,1:k];
    S0 = Diagonal(S[1:k]);
    W0 = W[:,1:k];
    L,Xs = eigen(V0'*H1*W0*inv(S0));
    X = V0[1:n,:]*Xs;
    inds = [];
    for i = 1:k
        X[:,i] = X[:,i]/norm(X[:,i])
        if norm(T(X[:,i],L[i])*X[:,i]) < tol
            push!(inds,i);
        end
    end
    return L[inds],X[:,inds];
end

end
