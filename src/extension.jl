# some extensions to ITensors.jl

getindex(T::ITensor, pairs::Tuple{Index, Int}...) = getindex(T, [IndexVal(pair...) for pair ∈ pairs]...)
getindex(T::ITensor, pairs::Vector{Tuple{Index, Int}}) = getindex(T, pairs...)

function contract(A::ITensor, B::ITensor, i::Index, j::Index)
    @assert i in inds(A)
    @assert j in inds(B)
    @assert dim(i) == dim(j)

    # Find the positions of the indices
    pos_i = findfirst(x -> x == i, inds(A))
    pos_j = findfirst(x -> x == j, inds(B))

    # Replace index i in A with j
    A_modified = replaceind(A, i => j)

    # Contract tensors
    C = A_modified * B
    return C
end

function clamp(A::ITensor, ind::Index, j::Int)
	v = zeros(dim(ind))
	v[j] = 1.0
	projector = ITensor(v, ind)
	return A*projector
end
