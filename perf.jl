using BenchmarkTools
using Krylov
using IncompleteLU

include("myownshit.jl")
include("optimizedshit.jl")

@inline function single(A, b, Pl)
    println("Reference")
    @btime bicgstab($A, $b, M=$Pl, ldiv=true)

    println("Old testin'")
    @btime mybicgstab($A, $b, $Pl)
    
    println("New testin'")
    @btime newbicgstab($A, $b, $Pl)
end


include("generators/symmetric-problem-fdm.jl")
include("generators/symmetric-problem-fem.jl")
include("generators/nonsymmetric-problem-fvm.jl")

@inline function singleprecon(name, t, f)
    println(name)
    A, b = t
    Pl = f(A)
    single(A, b, Pl)
    println()
end


singleprecon("fdm + I", fdmproblem(20), _ -> I)
singleprecon("fdm + ILU", fdmproblem(20), ilu)
singleprecon("fem + I", femproblem(20), _ -> I)
singleprecon("fem + ILU", femproblem(20), ilu)
singleprecon("fvm + ILU", fvmproblem(20), _ -> I)
singleprecon("fvm + ILU", fvmproblem(20), ilu)
