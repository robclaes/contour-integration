module PEPv #Contour Integration For Eigenvector Nonlinearities

using LinearAlgebra
using BlockArrays
using HomotopyContinuation


NodeType = Union{AbstractRange,Vector}


function getRandomPolynomial(x::Vector, d::Int)
    E,_ = exponents_coefficients(sum(x)^d,x)
    monomials = [prod(x.^E[:,i]) for i = 1:size(E,2)]
    poly = dot(randn(length(monomials)), monomials)
end

function getRandomMonomial(x::Vector, d::Int)
    E,_ = exponents_coefficients(sum(x)^d,x)
    monomials = [prod(x.^E[:,i]) for i = 1:size(E,2)]
    ind = round(Int,length(monomials)/2)
    mono = randn()*monomials[ind]
end


function getRandomPolynomialsFor(T::Matrix, x::Vector, z::Variable, nbcols::Int)
    n = size(T,1)
    T1 = T*ones(n);
    dT = [degree(T1[i],x) for i ∈ 1:n]
    reshape( [ getRandomPolynomial(x,dT[i]) for j ∈ 1:nbcols for i ∈ 1:n], n, nbcols)
end

function getRandomMonomialsFor(T::Matrix,x::Vector, z::Variable, nbcols::Int)
    n = size(T,1)
    T1 = T*ones(n)
    dT = [degree(T1[i],x) for i ∈ 1:n]
    reshape( [ getRandomMonomial(x,dT[i]) for j ∈ 1:nbcols for i ∈ 1:n], n, nbcols)
end


function getStartsols(S::System, z₀::Number)
    R = solve(S,target_parameters = [z₀],only_non_zero=true)
    return solutions(R)
end

function computeTrace_3(T::Matrix, x::Vector, z::Variable, z₀::Number, v::Vector, ϕ::Function, nodes::NodeType)
    S = System(T*x-v, parameters=[z])
    traces = zeros(ComplexF64, length(v),length(nodes) )
    startsols = getStartsols(S,z₀)
    traces[:,1] = sum(startsols)
    for i = 1:length(nodes)-1
        start_param = ϕ(nodes[i])
        end_param = ϕ(nodes[i+1])
        R = solve(S,startsols;start_parameters = [start_param], target_parameters = [end_param])
        startsols = solutions(R)
        traces[:,i+1] = sum(startsols)
    end
    return traces
end

function computeTrace_2(T::Matrix,x::Vector,z::Variable,z₀::Number,v::Vector,ϕ::Function,nodes::NodeType)
    S = System(T*x - v, parameters = [z])
    startsols = getStartsols(S, z₀)
    tracker = Tracker(ParameterHomotopy(S, [2.2], [2.2]))
    traces = zeros(ComplexF64, length(v),length(nodes) )
    traces[:,1] = sum(startsols)
    for i = 1:length(nodes)-1
        start_parameters!(tracker, [ϕ(nodes[i])])
        target_parameters!(tracker, [ϕ(nodes[i+1])])
        res = track.(tracker, startsols, 1.0, 0.0)
        startsols = [r.solution for r ∈ res]
        traces[:,i+1] = sum(startsols)
    end
    return traces
end

function computeTrace(T::Matrix, x::Vector, z::Variable, z₀::Number, v::Vector, ϕ::Function, nodes::NodeType)
    S = System(T*x - v, parameters = [z])
    startsols = getStartsols(S, z₀)
    R = solve(S,startsols;start_parameters = [z₀], target_parameters = [[ϕ(nodes[i])] for i = 1:length(nodes)])
    solution_sets = [solutions(RR[1]) for RR ∈ R]
    traces = [sum(solset) for solset ∈ solution_sets]
    return traces
end

function getTraceMatrices(T::Matrix, x::Vector, z::Variable, ϕ::Function, nodes::NodeType, V::Matrix)
    z₀ = ϕ(nodes[1])
    k = size(V,2)
    n = size(T,1)
    traceMatrices = zeros(ComplexF64,length(nodes),n,k)
    for i = 1:k
        v = V[:,i]
        traces = computeTrace_2(T,x,z,z₀,v,ϕ,nodes)
        for j = 1:length(nodes)
            traceMatrices[j,:,i] = traces[:,j]
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
    traceMatrices = getTraceMatrices(T, x, z, ϕ, nodes, V)
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
    res = []
    for i = 1:k
        X[:,i] = X[:,i]/norm(X[:,i])
        if norm(T(X[:,i],L[i])*X[:,i]) < tol
            push!(inds,i);
            push!(res,norm(T(X[:,i],L[i])*X[:,i]))
        end
    end
    return L[inds],X[:,inds],res;
end

end
