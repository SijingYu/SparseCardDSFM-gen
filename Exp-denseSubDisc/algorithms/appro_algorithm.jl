include("../../include/HyperLocal/Helper_Functions.jl")
#include("../../include/HyperLocal/maxflow.jl")
include("../../src/SparseCard.jl")
include("../../src/pwl_approx.jl")
include("utils.jl")

using MAT, StatsBase

function approx_gen_dense_sub(G::SparseMatrixCSC{Float64,Int64}, epsilon::Float64, exponent::Float64, to_exclude::Vector{Int64})
    println("=========================== Starting the algorithm ===========================")
    start = time()

    println("--------------------------- preparing the graph ---------------------------")
    N = size(G)[1]
    d_G = sum(G, dims=2)
    volG = sum(d_G)
    R = setdiff(collect(1:N), to_exclude)
    f_G = convex_f(N, exponent)

    BestS = R
    beta_old = Inf
    beta_new = DSGScore(G, R, exponent)
    betaBest = beta_new

    Edges = [collect(1:N)]
    EdgesW = [zeros(N+1)]
    #EdgesW[1][1] = betaBest*f_G

    counter = 1

    while betaBest < beta_old
        for i in 1:(N+1)
            EdgesW[1][i] = betaBest*(f_G - convex_f(i-1, exponent))
        end
        if !check_cb_submodular(EdgesW[1])
            return betaBest, EdgesW
        end
        #return Edges,EdgesW
        #return Edges, EdgesW, N
        A, s_red, t_red = AsymmetricCard_reduction(Edges, EdgesW, N, epsilon, false)
        #return A,s,t
        aux_num = size(A,1)-size(G,1)
        println("Size of the graph: $N nodes")
        println("Number of nodes to exclude: $(length(to_exclude))")
        println("Number of auxiliary nodes: $(aux_num)")
        println("Sparsity: $((size(A,1)-size(G,1))/(floor(N/2)+1))")
        A[1:N,1:N] =  A[1:N,1:N] + G
        A_shrink = shrink_graph(A, to_exclude)

        s = shrink_vec(s_red, to_exclude)
        t = shrink_vec(t_red, to_exclude)
        A_N = size(A_shrink)[1]
        d_A = sum(A_shrink, dims=2)
        volA = sum(d_A)
        println("Size of A shrink")
        println(A_N)

        iter_start = time()
        println("--------------------------- Iteration: $counter ---------------------------")
        println("Starting DSG Score: $beta_old")
        beta_old = betaBest
        println("Current BetaBest: $betaBest")

        sVec = ones(A_N) + s
        sVec[(end-aux_num):end] .= 0
        tVec = ones(A_N) + t
        tVec[(end-aux_num):end] .= 0
        tVec[end] = Inf

        #return A_shrink, sVec, tVec, s_red, t_red

        println("Starting maxflow")
        #println(A_N)
        #return A_shrink, sVec, tVec
        #return A_shrink,vec(sVec),tVec
        F = maxflow(A_shrink,vec(sVec),tVec,0)
        Src = source_nodes(F)[2:end].-1
        S_distorted = intersect(1:length(R), Src)
        println("Size of S distorted")
        println(length(S_distorted))

        if length(S_distorted) == 0
            time_used = time() - start
            println("=========================== Algorithm end ===========================")
            println("The algorithm uses $time_used")
            return BestS, betaBest
        end

        real_S = R[S_distorted]

        beta_new = DSGScore(G, real_S, exponent)
        println("New DSG Score: $beta_new")

        if beta_new < betaBest
            betaBest = beta_new
            BestS = real_S
            println("S updated to size of $(length(real_S))")
            counter = counter + 1
            iter_time_used = time() - iter_start
            println("This iteration uses $iter_time_used")
        end
    end
        time_used = time() - start
        println("=========================== Algorithm end here ===========================")
        println("The algorithm uses $time_used")
        return BestS, betaBest
end

function DSGScore(G::SparseMatrixCSC{Float64,Int64}, S::Vector{Int64}, alpha::Float64)
    cut, vol, edges, cond = set_stats(G, S, sum(G.nzval))
    N = size(G)[1]
    s_len = length(S)

    return cut/(convex_f(s_len, alpha))
end

function shrink_graph(G::SparseMatrixCSC{Float64,Int64}, to_exclude::Vector{Int64})
    N = size(G)[1]
    d = sum(G, dims=2)
    volG = sum(d)
    R = setdiff(collect(1:N), to_exclude)
    to_sum = sum(G[R, to_exclude], dims=2)
    A = sparse([to_sum;0.0])
    B = sparse(transpose(to_sum))
    return [[G[R,R];B] A]
end

function shrink_vec(whole::Vector{Float64}, to_exclude::Vector{Int64})
    N = length(whole)
    R = setdiff(collect(1:N), to_exclude)
    new_vec = whole[R]
    return [new_vec;0.0]
end

function convex_f(x::Int64, a::Float64)
    return x^a
end

