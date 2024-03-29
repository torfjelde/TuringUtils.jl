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


#####################################################
### Try to optimize the structure of the bijector ###
#####################################################
iscontiguous(left, right) = maximum(left) + 1 == minimum(right)

Bijectors.up1(b::Bijectors.Stacked) = b

optimize_bijector(b) = b
optimize_bijector(ib::Bijectors.Inverse) = inv(optimize_bijector(ib.orig))

function optimize_bijector(b::Bijectors.NamedBijector{names}) where {names}
    bs = map(b.bs) do b
        optimize_bijector(b)
    end
    return Bijectors.NamedBijector(NamedTuple{names}(bs))
end

# TODO: Maybe also handle the same cases as the evaluation of `TruncatedBijector`,
# e.g. `Bijectors.Log{N}() ∘ Bijectors.Scale{N}(-lb)` if `all(isfinite.(lb))` i.e.
# is lowerbounded.
function optimize_bijector(b::Bijectors.TruncatedBijector{N}) where {N}
    lb, ub = b.lb, b.ub
    return if all(iszero.(lb)) && all(isinf.(ub))
        Bijectors.Log{N}()
    elseif all(isfinite.(lb)) && all(isfinite.(ub))
        Bijectors.Logit{N}(lb, ub)
    else
        b
    end
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
            optimize_bijector(Bijectors.up1(b_r))
        else
            optimize_bijector(b_r)
        end
    end

    ranges = map(segments) do r
        minimum(b.ranges[minimum(r)]):maximum(b.ranges[maximum(r)])
    end

    @assert length(bs) == length(ranges) "number of bijectors and ranges is different"

    # Nvm; this actually causes issues for certain bijectors which we can't do `up1` to.
    # # If we're left with only a single bijector, don't wrap in `Stacked`.
    # length(bs) == 1 && return optimize_bijector(b.bs[1])

    return Bijectors.Stacked(bs, ranges)
end
