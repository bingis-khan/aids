using IterativeSolvers
using Krylov
using IncompleteLU
using BenchmarkTools
using LinearAlgebra

include("generators/symmetric-problem-fdm.jl")

A, b = fdmproblem(10)
LU = ilu(A)

println("IterativeSolvers:")
@btime bicgstabl(A, b, Pl=LU)
x = bicgstabl(A, b, Pl=LU)
println(norm(b - A * x))

println("Krylov:")
@btime bicgstab(A, b, M=LU, ldiv=true)
x = bicgstabl(A, b, Pl=LU)
println(norm(b - A * x))