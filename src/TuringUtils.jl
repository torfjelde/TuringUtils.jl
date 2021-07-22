module TuringUtils

import Bijectors

using DynamicPPL: DynamicPPL
using Turing: Turing
using MCMCChains: MCMCChains
using AbstractMCMC: AbstractMCMC

using Random: Random
using ProgressMeter: ProgressMeter

export NamedTupleChainIterator,
    # Functions
    fast_predict,
    fast_generated_quantities,
    # Modules
    MCMCChainsUtils,
    DynamicPPLUtils

include("packages/componentarrays.jl")
include("packages/bijectors.jl")
include("packages/dynamicppl.jl")
include("packages/mcmcchains.jl")

#######################################
# Prediction and generated quantities #
#######################################
"""
    fast_generated_quantities(model, chain)

Fast version of [`DynamicPPL.generated_quantities`](@ref) using `NamedTupleChainIterator`
together with [`DynamicPPLUtils.fast_setval_and_resample!!`](@ref) to achieve high performance.
"""
function fast_generated_quantities(model::DynamicPPL.Model, chain::DynamicPPL.AbstractChains)
    varinfo = DynamicPPL.VarInfo(model)
    iters = MCMCChainsUtils.NamedTupleChainIterator(
        keys(varinfo.metadata),
        chain,
        MCMCChainsUtils.getconverters(chain)
    )

    pm = ProgressMeter.Progress(length(iters))

    results = map(iters) do nt
        DynamicPPLUtils.fast_setval_and_resample!!(varinfo, nt)
        result = model(varinfo)

        ProgressMeter.next!(pm)

        return result
    end

    return reshape(results, size(chain, 1), size(chain, 3))
end

"""
    fast_predict(model, chain; kwargs...)

Fast version of [`Turing.Inference.predict`](@ref) using `NamedTupleChainIterator`
together with [`DynamicPPLUtils.fast_setval_and_resample!!`](@ref) to achieve high performance.
"""
function fast_predict(model::DynamicPPL.Model, chain::MCMCChains.Chains; kwargs...)
    return fast_predict(Random.GLOBAL_RNG, model, chain; kwargs...)
end
function fast_predict(
    rng::Random.AbstractRNG, model::DynamicPPL.Model, chain::MCMCChains.Chains;
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
    model::DynamicPPL.Model,
    chain::MCMCChains.Chains;
    sampler=DynamicPPL.SampleFromPrior()
)
    return fast_transitions_from_chain(Random.GLOBAL_RNG, model, chain; sampler)
end
function fast_transitions_from_chain(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    chain::MCMCChains.Chains;
    sampler=DynamicPPL.SampleFromPrior()
)
    varinfo = DynamicPPL.VarInfo(model)
    iters = MCMCChainsUtils.NamedTupleChainIterator(
        keys(varinfo.metadata),
        chain,
        MCMCChainsUtils.getconverters(chain)
    )

    pm = ProgressMeter.Progress(length(iters))

    transitions = map(iters) do nt
        # Set variables present in `chain` and mark those NOT present in chain to be resampled.
        DynamicPPLUtils.fast_setval_and_resample!!(varinfo, nt)
        model(rng, varinfo, sampler)

        # Convert `VarInfo` into `NamedTuple` and save.
        theta = DynamicPPL.tonamedtuple(varinfo)
        lp = Turing.getlogp(varinfo)

        ProgressMeter.next!(pm)

        Turing.Inference.Transition(theta, lp)
    end

    return reshape(transitions, size(chain, 1), size(chain, 3))
end


end # module
