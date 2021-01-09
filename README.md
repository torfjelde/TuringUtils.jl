# TuringUtils.jl
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

### Slightly more interesting example

```julia
julia> @model function demo2(x, N = 1)
           N = ismissing(x) ? N : length(x)
           s ~ InverseGamma(2, 3)
           x ~ filldist(Exponential(s), N)
       end
demo2 (generic function with 2 methods)

julia> model = demo2(missing, 10);

julia> varinfo = DynamicPPL.VarInfo(model);

julia> θ = ComponentArray(varinfo)
ComponentVector{Float64}(s = [0.8412747017520024], x = [0.9019541644418951, 0.42426023634478494, 0.8685025707534498, 0.36611637468075053, 0.13119408049163322, 0.7923953477085011, 0.2386958354461077, 0.1303953114230568, 0.6370253363450222, 0.07785344641161263])

julia> b = bijector(varinfo);

julia> b(θ)
ComponentVector{Float64}(s = [-0.1728370353018339], x = [-0.10319157568528073, -0.857408247062745, -0.14098473331290012, -1.004804032511878, -2.0310775216940202, -0.23269483531518204, -1.4325651926009966, -2.0371845854806496, -0.45094584971897594, -2.552927111783328])

julia> θ.x
10-element view(::Array{Float64,1}, 2:11) with eltype Float64:
 0.9019541644418951
 0.42426023634478494
 0.8685025707534498
 0.36611637468075053
 0.13119408049163322
 0.7923953477085011
 0.2386958354461077
 0.1303953114230568
 0.6370253363450222
 0.07785344641161263

julia> b(θ).x
10-element view(::Array{Float64,1}, 2:11) with eltype Float64:
 -0.10319157568528073
 -0.857408247062745
 -0.14098473331290012
 -1.004804032511878
 -2.0310775216940202
 -0.23269483531518204
 -1.4325651926009966
 -2.0371845854806496
 -0.45094584971897594
 -2.552927111783328
```
