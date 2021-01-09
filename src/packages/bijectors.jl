Base.names(::DynamicPPL.VarInfo{<:NamedTuple{names}}) where {names} = names
Base.names(::Type{<:DynamicPPL.VarInfo{<:NamedTuple{names}}}) where {names} = names


"""
    bijector(varinfo::DynamicPPL.VarInfo)

Returns a `NamedBijector` which can transform different variants of `varinfo`.

E.g. `ComponentArrays.ComponentArray(varinfo)`, `namedtuple(varinfo)`.
"""
@generated function Bijectors.bijector(varinfo::DynamicPPL.TypedVarInfo)
    names = Base.names(varinfo)
    
    expr = Expr(:tuple)
    for n in names
        e = :(Stacked2(map($(Bijectors).bijector, md.$n.dists), md.$n.ranges))
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

Base.inv(sb::Stacked) = Stacked2(map(Base.inv, sb.bs), sb.ranges)

function (sb::Stacked2{<:AbstractArray})(x::AbstractVector{<:Real})
    n = length(sb.ranges)
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
    return mapreduce(sum, 1:n) do i
        sum(logabsdetjac(b.bs[i], x[b.ranges[i]]))
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
