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

###############################
# Allow dispatching on models #
###############################
"""
    evaluatortype(f)
    evaluatortype(f, nargs)
    evaluatortype(f, argtypes)
    evaluatortype(m::DynamicPPL.Model)

Returns the evaluator-type for model `m` or a model-constructor `f`.

(!!!) If you're using Revise.jl, remember that you might need to re-instaniate
the model since `evaluatortype` might have changed.
"""
function evaluatortype(f, argtypes)
    rets = Core.Compiler.return_types(f, argtypes)
    if (length(rets) != 1) || !(first(rets) <: DynamicPPL.Model)
        error("inferred return-type of $(f) using $(argtypes) is not `Model`; please specify argument types")
    end
    # Extract the anonymous evaluator.
    return first(rets).parameters[1]
end
evaluatortype(f, nargs::Int) = evaluatortype(f, ntuple(_ -> Missing, nargs))
function evaluatortype(f)
    m = first(methods(f))
    # Extract the arguments (first element is the method itself).
    nargs = length(m.sig.parameters) - 1

    return evaluatortype(f, nargs)
end
evaluatortype(::DynamicPPL.Model{F}) where {F} = F

evaluator(m::DynamicPPL.Model) = m.f

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

end
