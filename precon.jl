using BenchmarkTools
using IterativeSolvers: cg
using LinearAlgebra
using Preconditioners


jacobiprecon(A) = diagm([1 / norm(A[:,i]) for i=1:size(A, 2)])


include("generators/symmetric-problem-fem.jl")


A = sprand(1000, 1000, 0.01)
A = A + A' + 30I
b = A*ones(1000)
precon = AMGPreconditioner{SmoothedAggregation}(A)


@btime cg(A, b, Pl=precon)
@btime cg(A, b)
