module CIFEN #Contour Integration For Eigenvector Nonlinearities

using LinearAlgebra
using BlockArrays

function blockHankel(Ai::Matrix{N} ...) where {N<:Number}
    @assert length(Ai)%2 == 0;
    
    K::Int = length(Ai)/2;
    n = size(Ai[1],1);
    
    if K == 1
        return Ai[1], Ai[2];
    end

    H0 = BlockArray{N}(undef_blocks,n*ones(Int,K),n*ones(Int,K));
    H1 = BlockArray{N}(undef_blocks,n*ones(Int,K),n*ones(Int,K));
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
    K = size(H0,1);
    V,S,W = svd(H0);
    # TODO: more robust rank test
    firstind = findfirst(S.<S[1]*length(S)*eps())
    k = isnothing(firstind) ? K : firstind - 1;
    V0 = V[:,1:k];
    S0 = Diagonal(S[1:k]);
    W0 = W[:,1:k];
    L,X = eigen(V0'*H1*W0*inv(S0));
    X = X[1:n,:];
    inds = [];
    for i = 1:k
        if norm(T(X[:,i],L[i])*X[:,i]) < tol
            push!(inds,i);
        end
    end
    return L[inds],X[:,inds];
end

end