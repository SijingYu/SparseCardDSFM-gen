# Strongly-local code for minimizing the HCL objective
# Implemented with the thresholded linear hyperedge splitting penalty.

include("Helper_Functions.jl")
include("maxflow.jl")
include("../../src/SparseCard.jl")

function CoreCut(G::SparseMatrixCSC{Float64,Int64},S::Vector{Float64}, tau::Float64, epsilon::Float64)
    
    N = size(G)[1]

    sVec = sum(G, dims=2)
    tVec = tau .* ones(N)

    Edges = [collect(1:N)]
    EdgesW = tau * collect(0:N) .* reverse(collect(0:N))/N
    A = SymmetricCard_reduction(Edges,EdgesW,N,epislon,returnIJV=true)
    Graph = G+A
    F = maxflow(A,sVec,tVec,0)
    Src = source_nodes_min(F)[2:end].-1
    S = intersect(1:n,Src)
    
end


function cut_part(G::SparseMatrixCSC{Float64,Int64}, tau::Float64)
    """
        accounts for cut(S)-vol(S)+tau|S|
    """
end