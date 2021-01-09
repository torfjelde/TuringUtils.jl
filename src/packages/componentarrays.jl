import ComponentArrays

# Constructor for `ComponentArray` from a `VarInfo`
function ComponentArrays.ComponentArray(varinfo::DynamicPPL.VarInfo)
    # TODO: this should give the right ordering, but maybe it would
    # be better to do the same as is done in `DynamicPPL.setall!` just to be sure?
    nt = map(DynamicPPL.tonamedtuple(varinfo)) do (vals, syms)
        vals
    end
    return ComponentArrays.ComponentArray(nt)
end

# Constructor for `ComponentArray` from a `DynamicPPL.Model`
function ComponentArrays.ComponentArray(varinfo::DynamicPPL.Model)
    varinfo = DynamicPPL.VarInfo(model)
    return ComponentArrays.ComponentArray(varinfo)
end

# Make it so that `NamedBijector` is compatible with `ComponentArray`
@generated function (b::Bijectors.NamedBijector{names1})(
    x::ComponentArrays.ComponentVector
) where {names1}
    ax = first(ComponentArrays.getaxes(x))
    indexmap = ComponentArrays.indexmap(ax)

    rs = Expr(:tuple)
    bs = []
    for (k, v) in pairs(indexmap)
        push!(rs.args, :($v))
        if k in names1
            push!(bs, :(b.bs[$(QuoteNode(k))]))
        else
            push!(bs, :(Identity{1}()))
        end
    end

    # Re-uses the implementation for `Stacked`, thus acting directly on the vector
    # rather than accessing the fields via the named fields.
    return quote
        y = $(Bijectors)._transform(ComponentArrays.getdata(x), $(rs), $(bs...))
        return $(ComponentArrays).ComponentArray(y, $(ax))
    end
end

# Just works, but impl in Bijectors.jl unnecessarily specializes on `x::NamedTuple`
@generated function Bijectors.logabsdetjac(b::Bijectors.NamedBijector{names}, x) where {names}
    exprs = [:($(Bijectors).logabsdetjac(b.bs.$n, x.$n)) for n in names]
    return :(+($(exprs...)))
end
