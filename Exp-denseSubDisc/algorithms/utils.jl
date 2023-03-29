using LightGraphs
using SparseArrays
using DelimitedFiles
#cite neighborhood from FlowSeed

# dictionry of edge(sorted nodes) to weight
function get_edges(A::SparseMatrixCSC{Float64,Int64})
    A_nz = findnz(A)
    edges = Dict()
    for i in 1:length(A_nz[1])
        edges[sort([A_nz[1][i],A_nz[2][i]])] = A_nz[3][i]
    end
    return edges
end

# matrix where each row is "first node, second node, edge weight"
function get_edges2(A::SparseMatrixCSC{Float64,Int64})
    A_nz = findnz(A)
    edges = Matrix{Float64}(undef, 3, 0)
    for i in 1:length(A_nz[1])
        #println(size(push!(Vector{Float64}(sort([A_nz[1][i],A_nz[2][i]])),A_nz[3][i])))
        #println(hcat(edges, push!(Vector{Float64}(sort([A_nz[1][i],A_nz[2][i]])),A_nz[3][i])))
        edges = hcat(edges, push!(Vector{Float64}(sort([A_nz[1][i],A_nz[2][i]])),A_nz[3][i]))
        #edges = vcat(edges, [sort([A_nz[1][i],A_nz[2][i]]),A_nz[3][i]])
    end
    return unique(transpose(edges),dims = 1)
end

function subgraph(A::SparseMatrixCSC{Float64,Int64}, set::Vector{Int64}, include_cut::Bool = false)
    n = size(A)[1]
    set_c = setdiff(1:n, set)
    set_len = length(set)
    if include_cut
        return sparse(vcat(fill(1:n,length(set))...,fill(set,(n-set_len))...),
            vcat((fill.(set,n)...),(fill.(set_c,set_len)...)), fill(1, (2*n-set_len)*set_len)).*A
    else
        if !(n in set)
            #=println(set_len)
            println(vcat(fill(set,set_len)...,[n]))
            println(vcat(fill.(set,set_len)...,[n]))
            println(vcat(fill(1,set_len),[0]))=#
            return sparse(vcat(fill(set,set_len)...,[n]),vcat(fill.(set,set_len)...,[n]),vcat(fill(1,set_len*set_len),[0])).*A
        else
            return sparse(vcat(fill(set,set_len)...),vcat(fill.(set,set_len)...),1).*A
        end
    end
end

function a_k(convex_f, k::Int64, n::Int64)
    if k < n 
        # positive due to convexity
        return convex_f(Float64(k)+1) + convex_f(Float64(k)-1) - 2*convex_f(Float64(k))
    else
        # negative for the a_n due to non-decreasing
        return convex_f(Float64(n)-1) - convex_f(Float64(n))
    end
end

function connected_subgraph(G::SparseMatrixCSC{Float64, Int64})
    connected = (d_G = vec(sum(G,dims = 2))).>0
    connected_nodes = deleteat!(collect(1:length(connected)).*connected, collect(1:length(connected)).*connected.==0)
    return G[connected_nodes, connected_nodes]
end

#=
function a_k_s(N_minus_r::Int64, convex_f)
    f_s = []
    for i in 0:(N_minus_r+1)
        if i % 1000000 == 0
            println(i)
        end
        push!(f_s, convex_f(Float64(i)))
    end
    #println("f_s done")
    aks = []
    for i in 1:(N_minus_r-1)
        push!(aks, f_s[i]+f_s[i+2]-2*f_s[i+1])
    end
    push!(aks, f_s[N_minus_r]-f_s[N_minus_r+1])
    return aks, f_s
end
=#

function a_k_s(N_minus_r::Int64, convex_f, a::Float64)
    f_s = []
    for i in 0:(N_minus_r+1)
       #if i % 1000000 == 0
        #    println(i)
        #end
        push!(f_s, convex_f(Float64(i), a))
    end
    #println("f_s done")
    aks = []
    for i in 1:(N_minus_r-1)
        push!(aks, round((f_s[i]+f_s[i+2]-2*f_s[i+1]); digits = 5))
    end
    push!(aks, f_s[N_minus_r]-f_s[N_minus_r+1])
    return Vector{Float64}(aks)
end

function cut_S(G::SparseMatrixCSC{Float64,Int64}, S::Vector{Int64})
    d_A = vec(sum(G, dims = 2))
    #println("done d_A")
    x = d_A[S]
    #println("done d_A[S]")
    return sum(x)-sum(G[S, S])
end

function read_set(filename::String)
    S = open(readdlm,filename)
    println(filename*" has length "*string(length(S)))
    return Vector{Int64}(vec(S))
end

function neighborhood(A::SparseMatrixCSC,R::Array{Int64},k::Int64)

    rp = A.rowval
    ci = A.colptr
    n = size(A,1)

    eS = zeros(n)
    eS[R] .= 1

    # For node i, the neighbors of i are rp[ci[i]:ci[i+1]-1]
    for i = R
        neighbs = rp[ci[i]:ci[i+1]-1]
        eS[neighbs] .= 1
    end

    # This could be more efficient, but recursively calling won't take too long
    # as long as k isn't too large
    if k == 1
        return findall(x->x!=0,eS)
    else
        return neighborhood(A,findall(x->x!=0,eS),k-1)
    end

end

function mat_partitioned_by_metis(matrix, k::Int64)
    indices = Metis.partition(Graph(matrix), k)
    groups = []
    for i in 1:k
        push!(groups,filter(!iszero, (indices .== i).* collect(1:length(indices))))
    end
    return groups
end


function log_metis_partition(matrix)
    n = size(matrix,1)
    log_2_n = log(2, n)
    groups = []
    complement_groups = []
    part = mat_partitioned_by_metis(matrix,2)
    for i in 1:(log_2_n-1)
        j = 0
        if rand() < 0.5
            j = 1
        else
            j = 2
        end
        push!(complement_groups, setdiff(collect(1:n),part[j]))
        push!(groups, part[j])
        part = mat_partitioned_by_metis(matrix[part[j],part[j]],2)
    end
    return groups, complement_groups
end

function convex_f(x::Float64, a::Float64)
    return x^a
end