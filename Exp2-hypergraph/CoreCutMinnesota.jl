# Strongly-local code for minimizing the HCL objective
# Implemented with the thresholded linear hyperedge splitting penalty.

include("Helper_Functions.jl")
include("maxflow.jl")
include("../../src/SparseCard.jl")

function CoreCut(G::SparseMatrixCSC{Float64,Int64},S::Vector{Float64}, tau::Float64, epsilon::Float64,
    to_exclude::Vector{Int64})
    
    N = size(G)[1]
    d = sum(G, dims=2)
    volG = sum(d)
    R = setdiff(collect(1:N), to_exclude)

    sVec = zeros(N)
    sVec = tau * ones(N) + d
    
    tVec = zeros(N)
    tVec[to_exclude] .= Inf

    Edges = [collect(1:N)]
    EdgesW = tau * collect(0:N) .* reverse(collect(0:N))/N

    A = SymmetricCard_reduction(Edges, EdgesW, N, epislon, returnIJV=true)
    A[1:N,1:N] =  A[1:N,1:N] + G

    BestS = R
    alpha_old = Inf
    alpha_new = CoreCutScore(G, R, tau)
    alphaBest = alpha_new

    while alphaBest < alpha_old
        alpha_old = alphaBest
        
        sVec = alphaBest*sVec
        F = maxflow(A,sVec,tVec,0)
        Src = source_nodes_min(F)[2:end].-1
        S = intersect(1:n,Src)
        alpha_new = CoreCutScore(G, S, tau)

        if alpha_new < alphaBest
            alphaBest = alpha_new
            BestS = S
        end
    end

    return BestS, alphaBest
end

function CoreCutScore(G::SparseMatrixCSC{Float64,Int64},S::Vector{Float64}, tau::Float64)
    cut, vol, edges, cond = set_stats(G, S, sum(G.nzval))
    N = size(G)[1]
    s_len = length(S)

    return  (cut+tau*s_len*(N-s_len)/N)/(vol+tau*s_len)
end