include("utils.jl")
include("../../include/HyperLocal/maxflow.jl")

using MAT, StatsBase, DelimitedFiles

function exact_algorithm(A::SparseMatrixCSC{Float64,Int64}, to_exclude::Vector{Int64}, exponent::Float64)
    N = size(A,1)
    r = length(to_exclude)
    ak_s = a_k_s(N-r, convex_f, exponent)
    #println("ak_s computed")
    #println(typeof(ak_s))

    S = setdiff(collect(1:N),to_exclude)
    #println("setdiff")
    #println(typeof(A))
    cut = cut_S(A, S)
    #println("cut")

    new_beta = cut/convex_f(Float64(length(S)), exponent)
    #println("Beta updated to "*string(new_beta))
    beta = Inf
    S_best = []

    while new_beta < beta
        S_best = S
        beta = new_beta

        #println("st_cut_step")
        st_flow, S = faster_st_cut_step_exclude_R_a_k(A,  to_exclude,  Vector{Float64}(ak_s), beta)
        #println("Set length with beta "*string(beta))
        #println(length(S))
        new_beta = cut_S(A, S)/convex_f(Float64(length(S)), exponent)
    end
    return beta, S_best
end

function faster_st_cut_step_exclude_R_a_k(A::SparseMatrixCSC{Float64,Int64}, to_exclude::Vector{Int64}, a_ks::Vector{Float64}, beta::Float64)

    N = size(A,1)
    q = length(to_exclude)

    d_A = vec(sum(A,dims = 2))

    aux_graph = spzeros(2*N-q-1,2*N-q-1)
    aux_graph[1:N,1:N] = aux_graph[1:N,1:N]+A #*a_k(convex_f,N,N)

    #println("1-N assigned in auxiliary graph")
    beta_aks = beta*a_ks
    #println("beta aks computed")
    ak_matrix = aux_sub_construction(N, beta_aks, to_exclude)
    #println("ak_matrix created")
    aux_graph[1:N, (N+1):(2*N-q-1)] = ak_matrix
    #println("upper right")
    aux_graph[(N+1):(2*N-q-1), 1:N] = (aux_graph[1:N,(N+1):(2*N-q-1)])'

    #println("finish auxiliary graph")
    sVec = zeros(2*N-q-1)
    tVec = zeros(2*N-q-1)

    sVec[1:N] = d_A.-2*beta*a_ks[N-q]
    #println("finish source node 1-N")

    tVec[1:N] = d_A .- beta*a_ks[N-q]
    tVec[to_exclude] .= Inf
    tVec[(N+1):end]=collect(1:(N-q-1)).*a_ks[1:(N-q-1)]

    #println("Begin maxflow")

    F = maxflow(aux_graph,sVec,tVec,0)
    Src = source_nodes(F)[2:end].-1
    S = intersect(1:(N),Src)
    
    return F,S
end

function merge_t_st_cut_step_exclude_R_a_k(A::SparseMatrixCSC{Float64,Int64}, to_exclude::Vector{Int64}, a_ks::Vector{Float64}, beta::Float64)
    N = size(A,1)
    R = setdiff(collect(1:N), to_exclude)
    r = length(R)

    d_A = vec(sum(A,dims = 2))
    d_R = d_A[R]

    vol_R_bar = sum(d_A[to_exclude]) 

    aux_graph = spzeros(2*r-1,2*r-1)
    aux_graph[1:r, 1:r] = A[R,R]

    println("1-N assigned in auxiliary graph")
    beta_aks = beta*a_ks
    println("beta aks computed")
    @time ak_matrix = merge_t_construction(r, beta_aks)

    sVec = zeros(2*r-1)
    tVec = zeros(2*r-1)

    sVec[1:r] = d_R .- 2*beta*a_ks[r]
    tVec[1:r] = d_R .- beta*a_ks[r] + vec(sum(A[to_exclude,R],dims=1))
    tVec[(r+1):end]=collect(1:(r-1)).*a_ks[1:(r-1)] 

    println("Begin maxflow")
    F = maxflow(aux_graph,sVec,tVec,0)
    Src = source_nodes(F)[2:end].-1
    S = intersect(1:(r),Src)
    
    return F, R[S]
end

function merge_t_algorithm(A::SparseMatrixCSC{Float64,Int64}, to_exclude::Vector{Int64}, start_beta::Float64, convex_f, a::Float64)
    N = size(A,1)
    r = length(to_exclude)
    ak_s = a_k_s(N-r, convex_f, a)

    beta = start_beta

    S = setdiff(collect(1:N),to_exclude)
    cut = cut_S(A, S)

    new_beta = cut/convex_f(Float64(length(S)), a)

    S_best = []

    while new_beta < beta
        S_best = S
        beta = new_beta
        println("============================start================================")
        st_flow, S = merge_t_st_cut_step_exclude_R_a_k(A,  to_exclude,  Vector{Float64}(ak_s), beta)
        new_beta = cut_S(A, S)/convex_f(Float64(length(S)),a)
        println("============================new beta================================")
        println(new_beta)
        println("====================================================================")
    end
    return beta, S_best
end


function aux_sub_construction(N, a_ks::Vector{Float64}, to_exclude)
    q = length(to_exclude)
    to_include = setdiff(collect(1:N), to_exclude)
    nzv = repeat(a_ks[1:(N-q-1)], inner = N-q)
    #println(length(nzv))
    rowv = repeat(to_include, N-q-1)
    #println(length(rowv))
    colp = [1:(N-q):((N-q-1)*(N-q)+1);]
    #println(length(colp))
    m = N
    n = N-q-1
    return SparseMatrixCSC(m,n,colp,rowv,nzv)
end

function merge_t_construction(r, a_ks::Vector{Float64})
    nzv = repeat(a_ks[1:(r-1)], inner = r)
    rowv = repeat(collect(1:r), r-1)
    colp = [1:r:((r-1)*(r)+1);]
    return SparseMatrixCSC(r,r-1,colp,rowv,nzv)
end

function convex_f(x::Int64, a::Float64)
    return x^a
end

