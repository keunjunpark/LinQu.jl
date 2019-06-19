
mutable struct QGate 
	data::Vector{T} where {T}
	pos::Vector{Int}
	function QGate(data,pos::Vector{Int}) #TODO: data specified as Vector{Number} breaks the code
		numqubit = Int(log2(length(data))/2)
		if numqubit != length(pos)
			error("A $(numqubit) qubit gate should act on $(numqubit) sites!")
		end
		new(data,pos)
	end
	QGate(data, pos::Int...) = QGate(data,_tuple_array(pos))
end # struct

pos(gate::QGate) = gate.pos
qubits(gate::QGate) = gate.pos # new
range(gate::QGate) = length(gate.pos)
gate_tensor(gate::QGate) = gate.data
data(gate::QGate) = gate.data # new
copy(gate::QGate) = QGate(copy(gate_tensor(gate)), copy(pos(gate)))

function checklocal(pos::Vector{Int})
	length(pos) == 1 && return true
	for i =2:length(pos)
		(pos[i] != pos[i-1]+1) && (return false)
	end
	return true
end
checklocal(qg::QGate) = checklocal(pos(qg))

function nonlocal_local(qg::QGate) #:: QGateSet
	#TODO : Could do optimization here
	position = pos(qg)
	length(position)>2 && error("currently only support 2 qubit gates\n")
	localgates = QGate[]
	left = min(position...)
	right = max(position...)
	#TODO: Now assume always move to the first index 
	while right > left+1
		push!(localgates, SwapGate(right-1, right))
		right-=1
	end
	swapback = reverse(localgates)
	localqg = movegate(qg,left, right)
	if position[1]>position[2]
		localqg = movegate(qg,right, left)
	end
	push!(localgates,localqg)
	localgates = vcat(localgates, swapback)
	return QGateSet(localgates)
end

function mult_nonlocal_local(qg::QGate)
	sorted = sort(pos(qg))
	perm = [1:1:sorted[end];]
	localgates = QGate[]
	for i =1: length(sorted)-1
		left = sorted[i]
		right = sorted[i+1]
		while(right>left+1)
			push!(localgates, SwapGate(right-1, right))
			(perm[right-1], perm[right]) = (perm[right], perm[right-1])
			right-=1
		end
		sorted[i+1] = sorted[i]+1
	end
	print("gates:", localgates)
	# swapback = reverse(localgates)
	# localqg = movegate(qg,)
end

function movegate!(qg::QGate, p::Vector{Int})
	if length(p) != range(qg)
		error(" Got wrong number of qubits to act on\n")
	end
	qg.pos = p
	return qg
end
movegate!(qg::QGate, p::Int...) = movegate!(qg, _tuple_array(p))
movegate(qg::QGate, p::Vector{Int}) = movegate!(copy(qg), p)
movegate(qg::QGate, p::Int...)= movegate!(copy(qg), p...)

# == sigle quibit gates == 
IGate(pos::Vector{Int}) = QGate(complex([1,0,0,1]),pos)
XGate(pos::Vector{Int}) = QGate(complex([0,1,1,0]),pos)
YGate(pos::Vector{Int}) = QGate(complex([0,1im,-1im,0]),pos)
ZGate(pos::Vector{Int}) = QGate(complex([1,0,0,-1]),pos)
TGate(pos::Vector{Int}) = QGate(complex([1,0,0,exp(π/4im)]),pos)
HGate(pos::Vector{Int}) = QGate(complex((1/√2)*[1,1,1,-1]),pos) # H could be decomposed in to XY / YZ gates
SGate(pos::Vector{Int}) = QGate(complex([1,0,0,1im]),pos)

IGate(pos::Int...) = IGate(_tuple_array(pos))
XGate(pos::Int...) = XGate(_tuple_array(pos))
YGate(pos::Int...) = YGate(_tuple_array(pos))
ZGate(pos::Int...) = ZGate(_tuple_array(pos))
TGate(pos::Int...) = TGate(_tuple_array(pos))
HGate(pos::Int...) = HGate(_tuple_array(pos)) # H could be decomposed in to XY / YZ gates
SGate(pos::Int...) = SGate(_tuple_array(pos))

TdagGate(pos::Vector{Int}) = QGate(complex([1,0,0,exp(-π/4im)]),pos)
TdagGate(pos::Int...) = TdagGate(_tuple_array(pos))
function Rx(θ, pos::Vector{Int})
	θ = float(θ)
	c = cos(θ/2.)
	s = -1im*sim(θ/2.)
	QGate(complex([c,s,s,c]),pos)
end

function Ry(θ, pos::Vector{Int})
	θ = float(θ)
	c = cos(θ/2)
	s = sin(θ/2)
	QGate([c,s,-s,c],pos)
end

function Rz(θ, pos::Vector{Int})
	θ = float(θ)
	exponent_ = θ/2
	QGate([exp(-exponent_),0,0,exp(exponent_)],pos)
end

Rx(θ::Number, pos::Int...) = Rx(θ, _tuple_array(pos))
Ry(θ::Number, pos::Int...) = Ry(θ, _tuple_array(pos))
Rz(θ::Number, pos::Int...) = Rz(θ, _tuple_array(pos))

# == two qubit gates ==
SwapGate(pos::Vector{Int}) = QGate(complex([1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1]),pos)
SwapGate(pos::Int...) = SwapGate(_tuple_array(pos))
CNOTGate(pos::Vector{Int}) = QGate(complex([1,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0]),pos)
CNOTGate(pos::Int...) = CNOTGate(_tuple_array(pos))
# CZGate() = QGate(Diagonal([1.,1.,1.,-1.]),2) # TODO: check diagonal working??
# CRGate(θ::Number) = QGate(Diagonal([1.,1.,1.,exp(1.0im*θ))]),2)
# CRkGate(k::Number) = CRGate()

# Toffoli 


# === new added====
ITensor(qg::QGate, inds::IndexSet) = ITensor(gate_tensor(qg), IndexSet(inds,prime(inds)))
ITensor(qg::QGate, ind::Index...) = ITensor(qg, IndexSet(_tuple_array(ind)))
# TODO: better way to recognize a swap gate?
isswap(qg::QGate) = (gate_tensor(qg)== complex([1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1]))
sameposition(A::QGate, B::QGate) = (sort(pos(A)) == sort(pos(B)))
repeatedswap(A::QGate, B::QGate) = (isswap(A) && isswap(B) && sameposition(A,B))

function show(io::IO, gate::QGate)
	print(pos(gate),"\n")
end