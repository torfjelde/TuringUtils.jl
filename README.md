# TuringUtils
My playground for hacks for Turing.jl and related packages.

## Current stuff
Now we can do stuff like:

```julia
julia> using Turing, ComponentArrays

julia> # Simple model
       @model function demo(x)
           s ~ InverseGamma(2, 3)
           m ~ Normal(0, √s)
           for i in eachindex(x)
               x[i] ~ Normal(m, √s)
           end
       end
demo (generic function with 1 method)

julia> model = demo(randn(100));

julia> θ = ComponentArrays.ComponentArray(varinfo)
ComponentVector{Float64}(s = [4.102402222089596], m = [-2.3328284005251367])

julia> b = Bijectors.bijector(varinfo) # X → ℝ
Bijectors.NamedBijector{(:s, :m),NamedTuple{(:s, :m),Tuple{Stacked2{Array{Bijectors.Log{0},1},Array{UnitRange{Int64},1}},Stacked2{Array{Identity{0},1},Array{UnitRange{Int64},1}}}}}((s = Stacked2{Array{Bijectors.Log{0},1},Array{UnitRange{Int64},1}}([Bijectors.Log{0}()], UnitRange{Int64}[1:1]), m = Stacked2{Array{Identity{0},1},Array{UnitRange{Int64},1}}([Identity{0}()], UnitRange{Int64}[1:1])))

julia> y, logjac = Bijectors.forward(b, θ)
(rv = (s = [1.4115727099600177], m = [-2.3328284005251367]), logabsdetjac = -1.4115727099600177)

julia> y
ComponentVector{Float64}(s = [1.4115727099600177], m = [-2.3328284005251367])

julia> logjac
-1.4115727099600177
```
