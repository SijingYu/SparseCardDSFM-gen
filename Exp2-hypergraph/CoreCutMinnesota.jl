include("../include/HyperLocal/Helper_Functions.jl")
include("../include/HyperLocal/maxflow.jl")
include("../src/SparseCard.jl")
using MatrixMarket
using StatsBase 

M = MatrixMarket.mmread("minnesota.mtx")
M_coord = MatrixMarket.mmread("minnesota_coord.mtx")
to_exclude = sample(collect(1:2642), 10,replace = false)
@CoreCut(SparseMatrixCSC{Float64, Int64}(M),0.05, 0.1,to_exclude)

function CoreCut(G::SparseMatrixCSC{Float64,Int64}, tau::Float64, epsilon::Float64,
    to_exclude::Vector{Int64})
    
    println("=========================== Starting the algorithm ===========================")
    start = time()

    println("--------------------------- preparing the graph ---------------------------")
    N = size(G)[1]
    d = sum(G, dims=2)
    volG = sum(d)
    R = setdiff(collect(1:N), to_exclude)

    Edges = [collect(1:N)]
    EdgesW = [(tau * collect(0:N) .* reverse(collect(0:N))/N)[1:Int64((floor(N/2)+1))]]
    A = SymmetricCard_reduction(Edges, EdgesW, N, epsilon, false)
    println("Size of the graph: $N nodes")
    println("Number of nodes to exclude: $(length(to_exclude))")
    println("Number of auxiliary nodes: $(size(A,1)-size(G,1))")
    println("Sparsity: $((size(A,1)-size(G,1))/(floor(N/2)+1))")
    A[1:N,1:N] =  A[1:N,1:N] + G
    A_N = size(A)[1]

    BestS = R
    alpha_old = Inf
    alpha_new = CoreCutScore(G, R, tau)
    alphaBest = alpha_new

    counter = 1
    while alphaBest < alpha_old
        iter_start = time()
        println("--------------------------- Iteration: $counter ---------------------------")
        println("Starting CoreCut Score: $alpha_old")
        alpha_old = alphaBest
        println("Current AlphaBest: $alphaBest")
        
        sVec = zeros(A_N)
        sVec[1:N] = tau * ones(N) + d
        sVec = alphaBest.*sVec

        tVec = zeros(A_N)
        tVec[to_exclude] .= Inf

        println("Staring maxflow")
        F = maxflow(A,vec(sVec),tVec,0)
        Src = source_nodes(F)[2:end].-1
        S = intersect(1:N, Src)

        if length(S) == 0
            time_used = time() - start
            println("=========================== Algorithm end ===========================")
            println("The algorithm uses $time_used")
            return BestS, alphaBest
        end
        alpha_new = CoreCutScore(G, S, tau)
        println("New CoreCut Score: $alpha_new")

        if alpha_new < alphaBest
            alphaBest = alpha_new
            BestS = S
            counter = counter + 1
            println("S updated to size of $(length(S))")
            iter_time_used = time() - iter_start
            println("This iteration uses $iter_time_used")
        end
    end

    println("=========================== Algorithm end ===========================")
    println("The algorithm uses $time_used")
    return BestS, alphaBest
end

function CoreCutScore(G::SparseMatrixCSC{Float64,Int64},S::Vector{Int64}, tau::Float64)
    cut, vol, edges, cond = set_stats(G, S, sum(G.nzval))
    N = size(G)[1]
    s_len = length(S)

    return  (cut+tau*s_len*(N-s_len)/N)/(vol+tau*s_len)
end