module PEPv #Contour Integration For Eigenvector Nonlinearities

using LinearAlgebra
using BlockArrays
using HomotopyContinuation
using SmithNormalForm

NodeType = Union{AbstractRange,Vector}

function getStartsols(S::System, z₀::Number)
    R = solve(S,target_parameters = [z₀])
    return solutions(R)
end

function reduceSolset(solset::Vector{Vector{T}} where T, p::Function)
    return sum([p(sol) for sol ∈ solset])
end

function reduceSolset(solset::Vector{Vector{T}} where T)
    return sum(solset)
end

function traceMonomials(T::Matrix,x::Vector,z::Variable)
    n = length(x)
    F = T*x
    FF = subs(F,z=>randn())
    allexp = fill(0,n,0)
    for ff in FF
        E, C = exponents_coefficients(ff,x)
        allexp = hcat(allexp,E)
    end
    SNF = smith(allexp)
    M = (SNF.S*diagm(SNF))[:,1:n]
    p = y -> [prod(y.^M[:,i]) for i = 1:n] 
    return p
end


function computeTrace(T::Matrix, x::Vector, z::Variable, z₀::Number, v::Vector, ϕ::Function, nodes::NodeType, p::Function)
    S = System(T*x - v, parameters = [z])
    startsols = getStartsols(S, z₀)
    R = solve(S,startsols;start_parameters = [z₀], target_parameters = [[ϕ(nodes[i])] for i = 1:length(nodes)])
    solution_sets = [solutions(RR[1]) for RR ∈ R]
    traces = [reduceSolset(solset, p) for solset ∈ solution_sets]
    return traces
end

function getTraceMatrices(T::Matrix, x::Vector, z::Variable, ϕ::Function, nodes::NodeType, V::Matrix, p::Function)
    z₀ = ϕ(nodes[1])
    k = size(V,2)
    n = size(T,1)
    traceMatrices = zeros(ComplexF64,length(nodes),n,k)
    for i = 1:k
        v = V[:,i]
        traces = computeTrace(T,x,z,z₀,v,ϕ,nodes,p)
        for j = 1:length(nodes)
            traceMatrices[j,:,i] = traces[j]
        end
    end
    traceMatrices = [traceMatrices[i,:,:] for i = 1:length(nodes)]
    return traceMatrices
end

function momentMatrix(traceMatrices::Vector{Matrix{T}} where T,nodes::NodeType, ϕ::Function, ϕprime::Function, i::Integer, trap::Bool=true)
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

function getMomentMatrices(T::Matrix, x::Vector, z::Variable, nodes::NodeType, ϕ::Function, ϕprime::Function, V::Matrix, highestMoment::Integer, trap::Bool=true)
    p = traceMonomials(T,x,z)
    traceMatrices = getTraceMatrices(T, x, z, ϕ, nodes, V, p)
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
