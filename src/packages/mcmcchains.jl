using MCMCChains: MCMCChains

maybe_unwrap_view(x) = x
maybe_unwrap_view(x::SubArray{<:Any,0}) = first(x)

"""
    NamedTupleChainIterator{names}(chains[, converters])

An iterator for `chains` returning `NamedTuple{names}` for each
iteration in `chains`, using `converters` to produce convert from
an iteration in a `MCMCChains.group` output into an `Array`.
"""
struct NamedTupleChainIterator{names,It}
    iter::It

    function NamedTupleChainIterator{names}(chains, converters=NamedTuple()) where {names}
        # 1. TODO: Flatten `Chains`.
        chain = chains

        # 2. Convert to `NamedTuple` of `Chains`.
        chain_nt = (; (
            (n, MCMCChains.group(chain, n)) for n in names
            if !isempty(MCMCChains.namesingroup(chain, n))
        )...);

        # TODO: Only use names that are present in the `chain`.
        new_names = keys(chain_nt)

        # 3. Convert the different variables according to `iter.converters`.
        nt_chain = convert_namedtuple_chain(chain_nt, converters)

        # 4. Make the underlying iterator.
        it = iterate_namedtuple(nt_chain, Val{1}())

        # 5. Convert 0-dimensional arrays into univariate.
        it_unwrapped = (map(maybe_unwrap_view, x) for x in it)

        return new{new_names,typeof(it_unwrapped)}(it_unwrapped)
    end
end

function NamedTupleChainIterator(names, chains, converters)
    return NamedTupleChainIterator{names}(chains, converters)
end
function NamedTupleChainIterator(names, chains; converters...)
    return NamedTupleChainIterator(names, chains, (; converters...))
end

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
Base.size(iter::NamedTupleChainIterator, dim) = Base.size(iter.iter, dim)
