Base.names(::DynamicPPL.VarInfo{<:NamedTuple{names}}) where {names} = names
Base.names(::Type{<:DynamicPPL.VarInfo{<:NamedTuple{names}}}) where {names} = names


"""
    bijector(varinfo::DynamicPPL.VarInfo)

Returns a `NamedBijector` which can transform different variants of `varinfo`.

E.g. `ComponentArrays.ComponentArray(varinfo)`, `namedtuple(varinfo)`.
"""
@generated function Bijectors.bijector(varinfo::DynamicPPL.TypedVarInfo; tuplify = false)
    names = Base.names(varinfo)
    
    expr = Expr(:tuple)
    for n in names
        e = quote
            if tuplify
                $(Bijectors.Stacked)(map($(Bijectors).bijector, tuple(md.$n.dists...)), md.$n.ranges)
            else
                $(Bijectors.Stacked)(map($(Bijectors).bijector, md.$n.dists), md.$n.ranges)
            end
        end
        push!(expr.args, e)
    end

    return quote
        md = varinfo.metadata
        bs = NamedTuple{$names}($expr)
        return $(Bijectors).NamedBijector(bs)
    end
end

# Relaxed `Bijectors.Stacked`
# TODO: Just relax the constraints on `Stacked` in the Bijectors.jl.
struct Stacked2{Bs, Rs} <: Bijectors.Bijector{1}
    bs::Bs
    ranges::Rs
end

Base.inv(sb::Stacked2) = Stacked2(map(Base.inv, sb.bs), sb.ranges)

function (sb::Stacked2{<:AbstractArray})(x::AbstractVector{<:Real})
    n = length(sb.ranges)
    n == 1 && return sb.bs[1](x[sb.ranges[1]])

    y = mapreduce(vcat, 1:n) do i
        sb.bs[i](x[sb.ranges[i]])
    end
    @assert size(y) == size(x) "x is size $(size(x)) but y is $(size(y))"
    return y
end

function Bijectors.logabsdetjac(
    b::Stacked2{<:AbstractArray},
    x::AbstractVector{<:Real}
)
    n = length(b.ranges)
    init = sum(Bijectors.logabsdetjac(b.bs[1], x[b.ranges[1]]))
    return mapreduce(+, 2:n; init = init) do i
        sum(Bijectors.logabsdetjac(b.bs[i], x[b.ranges[i]]))
    end
end

function Bijectors.logabsdetjac(b::Stacked2, x::AbstractMatrix{<:Real})
    return map(eachcol(x)) do c
        Bijectors.logabsdetjac(b, c)
    end
end

function Bijectors.forward(sb::Stacked2{<:AbstractArray}, x::AbstractVector)
    n = length(sb.ranges)
    yinit, linit = Bijectors.forward(sb.bs[1], x[sb.ranges[1]])

    n == 1 && return (rv = yinit, logabsdetjac = sum(linit))
    
    logjac = sum(linit)
    ys = Bijectors.mapvcat(drop(sb.bs, 1), drop(sb.ranges, 1)) do b, r
        y, l = forward(b, x[r])
        logjac += sum(l)
        y
    end
    return (rv = vcat(yinit, ys), logabsdetjac = logjac)
end


#####################################################
### Try to optimize the structure of the bijector ###
#####################################################
iscontiguous(left, right) = maximum(left) + 1 == minimum(right)

optimize_bijector(b) = b
function optimize_bijector(b::Bijectors.NamedBijector{names}) where {names}
    bs = map(b.bs) do b
        optimize_bijector(b)
    end
    return Bijectors.NamedBijector(NamedTuple{names}(bs))
end
function optimize_bijector(b::Bijectors.Stacked)
    n = length(b.ranges)
    segments = UnitRange[]
    start_idx = 1
    i = start_idx
    while start_idx <= n
        while i < n && iscontiguous(b.ranges[i], b.ranges[i + 1]) && b.bs[i] == b.bs[i + 1]
            i += 1
        end

        push!(segments, start_idx:i)
        i += 1
        start_idx = i
    end

    if b.bs isa Tuple
        segments = tuple(segments...)
    end

    bs = map(segments) do r
        b_r = b.bs[minimum(r)]
        if length(r) > 1 && Bijectors.dimension(b_r) == 0
            Bijectors.up1(b_r)
        else
            b_r
        end
    end

    ranges = map(segments) do r
        minimum(b.ranges[minimum(r)]):maximum(b.ranges[maximum(r)])
    end

    return Bijectors.Stacked(bs, ranges)
end
