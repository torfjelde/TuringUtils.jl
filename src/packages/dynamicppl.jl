module DynamicPPLUtils

using DynamicPPL

using Random: Random
using ProgressMeter: ProgressMeter

# Convenient overload
function DynamicPPL.set_flag!!(vi, vns::AbstractVector{<:VarName}, flag)
    new_vi = vi
    for vn in vns
        new_vi = DynamicPPL.set_flag!!(new_vi, vn, flag)
    end

    return new_vi
end

############################
# Simple utility functions #
############################
function replace_args(m::Model; kwargs...)
    return Model(m.name, m.f, deepcopy(merge(m.args, kwargs)), deepcopy(m.defaults))
end

#############################################
# Fast `setval!` and `setval_and_resample!` #
#############################################
"""
    fast_setval!!(vi; x...)
    fast_setval!!(vi, x::NamedTuple)

Return `vi` but now with values set according to `x`.

Internally this repeatedly calls [`fast_setval_inner!!`](@ref) repeatedly
on the metadata for each of the variables.

!!! Currently only `TypedVarInfo` is supported.
"""
fast_setval!!(vi::TypedVarInfo; kwargs...) = fast_setval!!(vi, (;kwargs...))
fast_setval!!(vi::TypedVarInfo, x::NamedTuple) = fast_setval!!(vi, vi.metadata, x)
@generated function fast_setval!!(
    vi::TypedVarInfo,
    metadata::NamedTuple{destnames},
    x::NamedTuple{srcnames},
) where {srcnames,destnames}
    # NOTE: This impl is only for `vi::TypedVarInfo`, hence everything will be mutating, i.e.
    # no need to keep track of the `vi`.
    expr = Expr(:block)
    for n in destnames
        # Only update those values present in `x`.
        if n in srcnames
            push!(expr.args, :(fast_setval_inner!!(vi, metadata.$n, x.$n)))
        end
    end
    # Return the `VarInfo`.
    push!(expr.args, :(return vi))

    return expr
end

function fast_setval_inner!!(vi::TypedVarInfo, md, val)
    vns = md.vns
    ranges = md.ranges
    vals = md.vals
    idcs = md.idcs

    for vn in vns
        idx = idcs[vn]
        r = ranges[idx]

        if hasindex(val, vn)
            # `_getindex` should be using `view`.
            vals[r] .= DynamicPPL._getindex(val, vn.indexing)
            DynamicPPL.settrans!!(vi, false, vn)
        end
    end

    return vi
end

"""
    fast_setval_and_resample!!(vi; x...)
    fast_setval_and_resample!!(vi, x::NamedTuple)

Return `vi` but now with values set according to `x`, enabling the
"del" flag for variable names _not_ found in `x`.

Internally this repeatedly calls [`fast_setval_and_resample_inner!!`](@ref)
repeatedly on the metadata for each of the variables.

!!! Currently only `TypedVarInfo` is supported.
"""
fast_setval_and_resample!!(vi::TypedVarInfo; kwargs...) = fast_setval_and_resample!!(vi, (;kwargs...))
fast_setval_and_resample!!(vi::TypedVarInfo, x::NamedTuple) = fast_setval_and_resample!!(vi, vi.metadata, x)
@generated function fast_setval_and_resample!!(
    vi::TypedVarInfo,
    metadata::NamedTuple{destnames},
    x::NamedTuple{srcnames},
) where {srcnames,destnames}
    # NOTE: This impl is only for `vi::TypedVarInfo`, hence everything will be mutating, i.e.
    # no need to keep track of the `vi`.
    expr = Expr(:block)
    for n in destnames
        # Only update those values present in `x`.
        if n in srcnames
            push!(expr.args, :(fast_setval_and_resample_inner!!(vi, metadata.$n, x.$n)))
        else
            push!(expr.args, :(DynamicPPL.set_flag!!(vi, metadata.$n.vns, "del")))
        end
    end
    # Return the `VarInfo`.
    push!(expr.args, :(return vi))

    return expr
end

function fast_setval_and_resample_inner!!(vi::TypedVarInfo, md, val)
    vns = md.vns
    ranges = md.ranges
    vals = md.vals
    idcs = md.idcs

    for vn in vns
        idx = idcs[vn]
        r = ranges[idx]

        if hasindex(val, vn)
            # `_getindex` should be using `view`.
            vals[r] .= DynamicPPL._getindex(val, vn.indexing)
            DynamicPPL.settrans!!(vi, false, vn)
        else
            # Set to be sampled.
            DynamicPPL.set_flag!!(vi, vn, "del")
        end
    end

    return vi
end

"""
    hasindex(val, vn::VarName)
    hasindex(val, indexing...)

Checks whether `vn`/`indexing` is indeed in `val`, i.e. not out-of-bounds.

# Examples
```jldoctest
julia> using DynamicPPL

julia> x = rand(2, 3);

julia> hasindex(x, @varname(x[1, 2]))
true

julia> hasindex(x, @varname(x[1]))
true

julia> hasindex(x, @varname(x[end]))
true

julia> hasindex(x, @varname(x[end + 1]))
false

julia> hasindex(x, @varname(x[:, end]))
true

julia> hasindex(x, @varname(x[:, end + 1]))
false

julia> hasindex(x, @varname(x[:, 0]))
false

julia> y = [x];

julia> hasindex(y, (1,), (1,:,:,:))
true

julia> hasindex(y, (1,), (1,:,2,:))
false

julia> hasindex(y, (:,), (1,:))
true

julia> hasindex(y, (:,), (1,:,2,:))
false
```
"""
hasindex(val, vn::VarName) = hasindex(val, vn.indexing...)
# Empty index.
hasindex(val) = true

# Cartesian indices.
function hasindex(val, index)
    indexp = ntuple(length(index)) do i
        if index[i] isa Colon
            # This even correctly covers cases such as
            # 
            #     x = rand(2); hasindex(x, (1, :, :, :))
            #
            # since the `:` in the tail will all result in `true`.
            true
        else
            # Covers cases such as `[1, 3]`, `1:2`, `1`.
            all(index[i] .∈ Ref(axes(val, i)))
        end
    end

    return all(indexp)
end
function hasindex(val, index, indexing...)
    return hasindex(val, index) && hasindex(view(val, index...), indexing...)
end

# Linear indexing.
hasindex(val, index::Tuple{Colon}) = true
hasindex(val, index::Tuple{Int}) = first(index) ≤ length(val)
hasindex(val, index::Tuple{<:AbstractVector{Int}}) = all(first(index) .≤ length(val))

function hasindex(val, index::Tuple{<:Any}, indexing...)
    i = first(index)
    return hasindex(val, index) && hasindex(view(val, CartesianIndices(val)[i]), indexing...)
end

#######################################
# Prediction and generated quantities #
#######################################
"""
    fast_generated_quantities(model, chain)

Fast version of [`DynamicPPL.generated_quantities`](@ref) using `NamedTupleChainIterator`
together with [`fast_setval_and_resample!!`](@ref) to achieve high performance.
"""
function fast_generated_quantities(model::DynamicPPL.Model, chain::DynamicPPL.AbstractChains)
    varinfo = DynamicPPL.VarInfo(model)
    iters = NamedTupleChainIterator(keys(varinfo.metadata), chain, getconverters(chain))

    pm = ProgressMeter.Progress(length(iters))

    results = map(iters) do nt
        TuringUtils.fast_setval_and_resample!!(varinfo, nt)
        result = model(varinfo)

        ProgressMeter.next!(pm)

        return result
    end

    return reshape(results, size(chain, 1), size(chain, 3))
end

"""
    fast_predict(model, chain; kwargs...)

Fast version of [`Turing.Inference.predict`](@ref) using `NamedTupleChainIterator`
together with [`fast_setval_and_resample!!`](@ref) to achieve high performance.
"""
function fast_predict(model::Model, chain::MCMCChains.Chains; kwargs...)
    return fast_predict(Random.GLOBAL_RNG, model, chain; kwargs...)
end
function fast_predict(
    rng::Random.AbstractRNG, model::Model, chain::MCMCChains.Chains;
    include_all = false
)
    # Don't need all the diagnostics
    chain_parameters = MCMCChains.get_sections(chain, :parameters)

    spl = DynamicPPL.SampleFromPrior()

    # Sample transitions using `spl` conditioned on values in `chain`
    transitions = fast_transitions_from_chain(rng, model, chain_parameters; sampler = spl)

    # Let the Turing internals handle everything else for you
    chain_result = reduce(
        MCMCChains.chainscat, [
            AbstractMCMC.bundle_samples(
                transitions[:, chain_idx],
                model,
                spl,
                nothing,
                MCMCChains.Chains
            ) for chain_idx = 1:size(transitions, 2)
        ]
    )

    parameter_names = if include_all
        names(chain_result, :parameters)
    else
        filter(k -> ∉(k, names(chain_parameters, :parameters)), names(chain_result, :parameters))
    end

    return chain_result[parameter_names]
end

function fast_transitions_from_chain(
    model::Turing.Model,
    chain::MCMCChains.Chains;
    sampler = DynamicPPL.SampleFromPrior()
)
    return fast_transitions_from_chain(Random.GLOBAL_RNG, model, chain; sampler)
end
function fast_transitions_from_chain(
    rng::Random.AbstractRNG,
    model::Turing.Model,
    chain::MCMCChains.Chains;
    sampler = DynamicPPL.SampleFromPrior()
)
    varinfo = Turing.VarInfo(model)
    iters = NamedTupleChainIterator(keys(varinfo.metadata), chain, getconverters(converters))

    pm = ProgressMeter.Progress(length(iters))

    transitions = map(iters) do nt
        # Set variables present in `chain` and mark those NOT present in chain to be resampled.
        TuringUtils.fast_setval_and_resample!!(varinfo, nt)
        model(rng, varinfo, sampler)

        # Convert `VarInfo` into `NamedTuple` and save.
        theta = DynamicPPL.tonamedtuple(varinfo)
        lp = Turing.getlogp(varinfo)

        ProgressMeter.next!(pm)

        Turing.Inference.Transition(theta, lp)
    end

    return reshape(transitions, size(chain, 1), size(chain, 3))
end

end
