Base.names(::DynamicPPL.VarInfo{<:NamedTuple{names}}) where {names} = names
Base.names(::Type{<:DynamicPPL.VarInfo{<:NamedTuple{names}}}) where {names} = names

Base.@kwdef struct BijectorStructureOptions
    static_parameters::Bool=true
    static_structure::Bool=true
    unvectorize_univariates::Bool=false
end

unvectorize_univariates(options::BijectorStructureOptions) = options.unvectorize_univariates
use_static_parameters(options::BijectorStructureOptions) = options.static_parameters
use_static_structure(options::BijectorStructureOptions) = options.static_structure

function Bijectors.NamedBijector(model::DynamicPPL.Model)
    return Bijectors.NamedBijector(DynamicPPL.VarInfo(model))
end

function Bijectors.NamedBijector(varinfo::DynamicPPL.TypedVarInfo)
    md = varinfo.metadata
    bs = map(bijector_from_metadata, md)
    return Bijectors.NamedBijector(bs)
end

function bijector_from_metadata(md::DynamicPPL.Metadata)
    b = Bijectors.Stacked(map(Bijectors.bijector, md.dists), md.ranges)
    return length(md.dists) == 1 ? first(b.bs) : b
end

function Bijectors.forward(
    b::Bijectors.NamedBijector,
    vi::DynamicPPL.SimpleVarInfo{<:NamedTuple}
)
    vals, logjac = Bijectors.forward(b, vi.values)
    return DynamicPPL.Setfield.@set(vi.values = vals), logjac
end

function unvectorize_univariate_maybe(md::DynamicPPL.Metadata, b::Bijectors.Stacked)
    # Number of dists should be 1 and that should be a univariate.
    (length(md.dists[1]) == 1 && md.dists[1] isa Bijectors.UnivariateDistribution) || return b
    # Number of bijectors and ranges should be 1.
    (length(b.bs) == 1 && length(first(b.ranges)) == 1) || return b
    # We only want to unvectorize if it's indeed represented by a symbol,
    # not some indexing expression, e.g. `x` is cool, `x[1]` is not.
    (length(md.vns) == 1 && md.vns[1].indexing === ()) || return b

    return first(b.bs)
end

#####################################################
### Try to optimize the structure of the bijector ###
#####################################################
iscontiguous(left, right) = maximum(left) + 1 == minimum(right)

Bijectors.up1(b::Bijectors.Stacked) = b

optimize_bijector_structure(b; options=BijectorStructureOptions()) = b
optimize_bijector_structure(ib::Bijectors.Inverse; kwargs...) = inv(optimize_bijector_structure(ib.orig; kwargs...))

function optimize_bijector_structure(b::Bijectors.NamedBijector{names}; kwargs...) where {names}
    bs = map(b.bs) do b
        optimize_bijector_structure(b; kwargs...)
    end
    return Bijectors.NamedBijector(NamedTuple{names}(bs))
end

# TODO: Maybe also handle the same cases as the evaluation of `TruncatedBijector`,
# e.g. `Bijectors.Log{N}() ∘ Bijectors.Scale{N}(-lb)` if `all(isfinite.(lb))` i.e.
# is lowerbounded.
function optimize_bijector_structure(
    b::Bijectors.TruncatedBijector{N};
    options=BijectorStructureOptions()
) where {N}
    # If we're not assuming static parameters, then there's nothing we can do here
    # and so we just return immediately.
    if !use_static_parameters(options)
        return b
    end

    lb, ub = b.lb, b.ub
    return if all(iszero.(lb)) && all(isinf.(ub))
        Bijectors.Log{N}()
    elseif all(isfinite.(lb)) && all(isfinite.(ub))
        Bijectors.Logit{N}(lb, ub)
    else
        b
    end
end

function optimize_bijector_structure(b::Bijectors.Stacked; options=BijectorStructureOptions())
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

    bs = map(segments) do r
        b_r = b.bs[minimum(r)]
        if length(r) > 1 && Bijectors.dimension(b_r) == 0
            optimize_bijector_structure(Bijectors.up1(b_r))
        else
            optimize_bijector_structure(b_r)
        end
    end

    ranges = map(segments) do r
        minimum(b.ranges[minimum(r)]):maximum(b.ranges[maximum(r)])
    end

    @assert length(bs) == length(ranges) "number of bijectors and ranges is different"

    # `static_structure` means that we want to tuplify the entire thing.
    if use_static_structure(options)
        ranges = tuple(ranges...)
        bs = tuple(bs...)
    end

    return Bijectors.Stacked(bs, ranges)
end
