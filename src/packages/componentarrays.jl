import ComponentArrays

"""
    namedtuple(vi::DynamicPPL.TypedVarInfo)

Similar to `DynamicPPL.tonamedtuple` but values of return-value are only the values
of the different variables rather than a tuple of values and symbols.
"""
namedtuple(vi::DynamicPPL.TypedVarInfo) = _namedtuple(vi.metadata)
@generated function _namedtuple(metadata::NamedTuple{names}, start = 0) where {names}
    expr = Expr(:tuple)
    start = :(1)
    for f in names
        length = :(length(metadata.$f.vals))
        push!(expr.args, :(metadata.$f.vals[1:$length]))
    end
    return :(NamedTuple{$names}($expr))
end

# Constructor for `ComponentArray` from a `VarInfo`
function ComponentArrays.ComponentArray(varinfo::DynamicPPL.VarInfo)
    return _tocomponentarray(varinfo.metadata)
end

function ComponentArrays.ComponentArray(varinfo::DynamicPPL.Model)
    varinfo = DynamicPPL.VarInfo(model)
    return ComponentArrays.ComponentArray(varinfo)
end

@generated function _tocomponentarray(metadata::NamedTuple{names}) where {names}
    # Adapted from https://github.com/TuringLang/DynamicPPL.jl/blob/873d5e94fe778514d52e24338a3cd1bd0535eefa/src/varinfo.jl#L338-L348
    # and thus ought should allow usage such as
    #
    #    varinfo[sampler] .= ComponentArrays.getdata(ComponentArray(varinfo))
    #
    # i.e. have the same order as `varinfo[sampler]`.
    ranges = Expr(:tuple)
    expr = Expr(:vcat)
    start = :(1)
    for f in names
        length = :(length(metadata.$f.vals))
        finish = :($start + $length - 1)
        r = :($start:$finish)
        push!(ranges.args, r)
        push!(expr.args, :(metadata.$f.vals[1:$length]))
        start = :($start + $length)
    end
    
    return quote
        ax = $(ComponentArrays).Axis(NamedTuple{$names}($ranges))
        return $(ComponentArrays).ComponentArray($expr, ax)
    end
end

# Make it so that `NamedBijector` is compatible with `ComponentArray`
vec_mby(x) = x
vec_mby(x::Real) = [x]
vec_mby(x::AbstractArray) = vec(x)

@generated function (b::Bijectors.NamedBijector{names1})(
    x::ComponentArrays.ComponentVector
) where {names1}
    ax = first(ComponentArrays.getaxes(x))
    indexmap = ComponentArrays.indexmap(ax)

    expr = Expr(:call, :vcat)
    for (k, v) in pairs(indexmap)
        if k in names1
            push!(expr.args, :(vec_mby(b.bs.$k(x.$k))))
        else
            push!(expr.args, :(vec_mby(x.$k)))
        end
    end

    # Re-uses the implementation for `Stacked`, thus acting directly on the vector
    # rather than accessing the fields via the named fields.
    return :(ComponentArrays.ComponentArray($expr, $ax))
end


# Just works, but impl in Bijectors.jl unnecessarily specializes on `x::NamedTuple`
@generated function Bijectors.logabsdetjac(b::Bijectors.NamedBijector{names}, x) where {names}
    exprs = [:($(Bijectors).logabsdetjac(b.bs.$n, x.$n)) for n in names]
    return :(+($(exprs...)))
end
