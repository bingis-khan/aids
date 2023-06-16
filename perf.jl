using BenchmarkTools
using IncompleteLU

include("myownshit.jl")
include("optimizedshit.jl")

function single(A, b, Pl)
    println("Old testin'")
    @btime mybicgstab(A, b, Pl)
    
    println("New testin'")
    @btime newbicgstab(A, b, Pl)
end


include("generators/symmetric-problem-fem.jl")

A, b = femproblem(10)
Pl = ilu(A)


single(A, b, Pl)