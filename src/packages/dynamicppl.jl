module DynamicPPLUtils

using DynamicPPL
using DynamicPPL: Setfield

using Random: Random
using ProgressMeter: ProgressMeter

maybevec(x) = x
maybevec(x::AbstractArray) = vec(x)

############################
# Simple utility functions #
############################
replace_args(m::Model; kwargs...) = Setfield.@set(m.args = deepcopy(merge(m.args, kwargs)))

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

        # Get the `Setfield.Lens` used within `vn`.
        lens = DynamicPPL.getlens(vn)
        if DynamicPPL.canview(lens, val)
            vals[r] .= maybevec(Setfield.get(val, lens))
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

        # Get the `Setfield.Lens` used within `vn`.
        lens = DynamicPPL.getlens(vn)
        if DynamicPPL.canview(lens, val)
            vals[r] .= maybevec(Setfield.get(val, lens))
            DynamicPPL.settrans!!(vi, false, vn)
        else
            # Set to be sampled.
            DynamicPPL.set_flag!(vi, vn, "del")
        end
    end

    return vi
end

# Hotfix for https://github.com/TuringLang/DynamicPPL.jl/issues/386.
# Overload "empty" `TypedVarInfo`.
function Base.eltype(
    vi::DynamicPPL.VarInfo{NamedTuple{(),Tuple{}}},
    spl::Union{DynamicPPL.AbstractSampler,DynamicPPL.SampleFromPrior}
)
    return eltype(DynamicPPL.getlogp(vi))
end

end
