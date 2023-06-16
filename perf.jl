using BenchmarkTools
using Krylov
using IncompleteLU

include("myownshit.jl")
include("optimizedshit.jl")

function single(A, b, Pl)
    println("Reference")
    @btime bicgstab(A, b, M=Pl, ldiv=true)

    println("Old testin'")
    @btime mybicgstab(A, b, Pl)
    
    println("New testin'")
    @btime newbicgstab(A, b, Pl)
end


include("generators/symmetric-problem-fdm.jl")

A, b = fdmproblem(20)
Pl = ilu(A)


single(A, b, Pl)