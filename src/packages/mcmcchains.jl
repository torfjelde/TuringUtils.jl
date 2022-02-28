module MCMCChainsUtils

using MCMCChains: MCMCChains
using DynamicPPL: DynamicPPL

export NamedTupleChainIterator

#####################################
# Working with groups of parameters #
#####################################
function get_groups(chains)
    parameters = MCMCChains.get_sections(chains, :parameters)
    # The last `.` in the key corresponds to the end of the possible prefix.
    groups = map(keys(parameters)) do k
        s = string(k)
        dotseps = split(s, ".")
        dotseps[end] = first(split(dotseps[end], "["))
        join(dotseps, ".")
    end
    unique_groups = unique(groups)
    return map(Symbol, unique_groups)
end

# TODO: Implement improved `MCMCChains.group` method which accounts
# for possible prefixes.

#################################
# Adding converters to `Chains` #
#################################
"""
    getconverters(chain)

Return `chain.info.converters` if the field exists, otherwise
`NamedTuple()` is returned.
"""
function getconverters(chain::MCMCChains.Chains)
    return if haskey(chain.info, :converters)
        chain.info.converters
    else
        NamedTuple()
    end
end

"""
    setconverters(chain::MCMCChains.Chains; converters...)
    setconverters(chain::MCMCChains.Chains, converters::NamedTuple)

Set `chain.info.converter` to `converters` and return the resulting `chain`.
"""
setconverters(chain::MCMCChains.Chains; converters...) = setconverters(chain, (; converters...))
function setconverters(chain::MCMCChains.Chains, converters)
    return MCMCChains.setinfo(chain, merge(chain.info, (converters=converters, )))
end

"""
    setconverters(chain::MCMCChains.Chains, model::DynamicPPL.Model)

Set `chain.info.converter` to the converters corresponding to `model`,
assuming the `:parameter` section in `chain` corresponds to a chain
obtained from sampling fom `model`.

This needs to be implemented on a per-model basis, i.e. there's no
default implementation. To correctly dispatch on a model without
instantiation, see [`DynamicPPLUtils.evaluatortype`](@ref).

(!!!) Ordering of variables matters! Hence the converters are not guaranteed
to produce the correct results for a `chain` with the same parameters as those
which would have been produced by `model` unless the ordering of the variables is
also correct.
"""
setconverters(chain::MCMCChains.Chains, model::DynamicPPL.Model) = chain

#############################
# `NamedTupleChainIterator` #
#############################
maybe_unwrap_view(x) = x
maybe_unwrap_view(x::SubArray{<:Any,0}) = first(x)

function convert_namedtuple_chain(chain_nt; converters...)
    return convert_namedtuple_chain(chain_nt, (; converters...))
end
@generated function convert_namedtuple_chain(
    chain_nt::NamedTuple{names},
    converters::NamedTuple{namesconv},
    default_converter=Array
) where {names, namesconv}
    vals = Expr(:tuple)
    for n in names
        if n in namesconv
            push!(vals.args, :(converters.$n(chain_nt.$n)))
        else
            push!(vals.args, :(default_converter(chain_nt.$n)))
        end
    end

    return :(NamedTuple{$names}($vals))
end

@generated function iterate_namedtuple(nt::NamedTuple{names}, ::Val{dim}) where {names, dim}
    iterators = []
    for n in names
        push!(iterators, :(eachslice(nt.$n, dims=$dim)))
    end

    return :(Iterators.map(NamedTuple{$names}, zip($(iterators...))))
end


"""
    NamedTupleChainIterator{names}(chains[, converters])

An iterator for `chains` returning `NamedTuple{names}` for each
iteration in `chains`, using `converters` to produce convert from
an iteration in a `MCMCChains.group` output into an `Array`.

# Example
```julia
julia> using Turing, TuringUtils

julia> Turing.setprogress!(false);
[ Info: [Turing]: progress logging is disabled globally
[ Info: [AdvancedVI]: global PROGRESS is set as false

julia> @model function demo(x)
           s ~ InverseGamma(2, 3)
           m ~ Normal(0, √s)
           for i in eachindex(x)
               x[i] ~ Normal(m, √s)
           end
       end
demo (generic function with 1 method)

julia> m = demo([1.0]);

julia> chain = sample(m, Prior(), 1_000)
Chains MCMC chain (1000×3×1 Array{Float64, 3}):

Iterations        = 1:1:1000
Number of chains  = 1
Samples per chain = 1000
Wall duration     = 0.07 seconds
Compute duration  = 0.07 seconds
parameters        = m, s
internals         = lp

Summary Statistics
  parameters      mean       std   naive_se      mcse        ess      rhat   ess_per_sec 
      Symbol   Float64   Float64    Float64   Float64    Float64   Float64       Float64 

           s    2.8537    5.7286     0.1812    0.1942   953.6235    1.0037    13244.7712
           m    0.0012    1.5694     0.0496    0.0438   822.4693    1.0015    11423.1847

Quantiles
  parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
      Symbol   Float64   Float64   Float64   Float64   Float64 

           s    0.5318    1.1159    1.7019    2.9830   11.8815
           m   -3.2980   -0.8438    0.0134    0.7904    3.0848

julia> it = Iterators.Stateful(NamedTupleChainIterator((:s, :m), chain));

julia> first(it)
(s = 2.2006853452070634, m = -1.040183332610299)

julia> NamedTuple{(:s, :m)}((Array(chain[1, [:s, :m], 1])...,))
(s = 2.2006853452070634, m = -1.040183332610299)

julia> first(it)
(s = 9.335656604810142, m = -2.3860978886120128)

julia> NamedTuple{(:s, :m)}((Array(chain[2, [:s, :m], 1])...,))
(s = 9.335656604810142, m = -2.3860978886120128)

julia> first(it)
(s = 0.35560679541085854, m = -0.8032064411702172)

julia> NamedTuple{(:s, :m)}((Array(chain[3, [:s, :m], 1])...,))
(s = 0.35560679541085854, m = -0.8032064411702172)
```
"""
struct NamedTupleChainIterator{names,It}
    iter::It

    function NamedTupleChainIterator{names}(chains, converters=NamedTuple()) where {names}
        # 1. Flatten `Chains`.
        chain = size(chains, 3) > 1 ? MCMCChains.pool_chain(chains) : chains

        # 2. Convert to `NamedTuple` of `Chains`.
        chain_nt = (; (
            (n, MCMCChains.group(chain, n)) for n in names
            if !isempty(MCMCChains.namesingroup(chain, n))
        )...);

        # 3. Only use names that are present in the `chain`.
        new_names = keys(chain_nt)

        # 4. Convert the different variables according to `iter.converters`.
        nt_chain = convert_namedtuple_chain(chain_nt, converters)

        # 5. Make the underlying iterator.
        # `MCMCChains.Chains` uses the first dimension as the iteration-dimension.
        it = iterate_namedtuple(nt_chain, Val{1}())

        # 6. Convert 0-dimensional arrays into univariate.
        it_unwrapped = (map(maybe_unwrap_view, x) for x in it)

        return new{new_names,typeof(it_unwrapped)}(it_unwrapped)
    end
end

function NamedTupleChainIterator(chains; converters...)
    NamedTupleChainIterator((get_groups(chains)...,), chains; converters...)
end
function NamedTupleChainIterator(names, chains; converters...)
    return NamedTupleChainIterator(names, chains, (; converters...))
end
function NamedTupleChainIterator(names, chains, converters::NamedTuple{()})
    return NamedTupleChainIterator{names}(chains, getconverters(chains))
end
function NamedTupleChainIterator(names, chains, converters)
    return NamedTupleChainIterator{names}(chains, converters)
end


Base.keys(::NamedTupleChainIterator{names}) where {names} = names

# Defer iteration interface to `.iter`.
Base.iterate(iter::NamedTupleChainIterator{()}) = nothing
Base.iterate(iter::NamedTupleChainIterator) = Base.iterate(iter.iter)

function Base.iterate(iter::NamedTupleChainIterator, state)
    return Base.iterate(iter.iter, state)
end

Base.IteratorSize(::NamedTupleChainIterator{<:Any,It}) where {It} = Base.IteratorSize(It)
Base.IteratorEltype(::NamedTupleChainIterator{<:Any,It}) where {It} = Base.IteratorEltype(It)
Base.eltype(::NamedTupleChainIterator{<:Any,It}) where {It} = Base.eltype(It)
Base.length(iter::NamedTupleChainIterator) = Base.length(iter.iter)
Base.size(iter::NamedTupleChainIterator) = Base.size(iter.iter)
Base.size(iter::NamedTupleChainIterator, dim) = Base.size(iter.iter, dim)

end
