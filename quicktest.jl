
include("generators/symmetric-problem-fdm.jl")


A, b = fdmproblem(6)

using Krylov
using LinearAlgebra

ksol, = bicgstab(A, b)

kdif = b - A * ksol
println("solution: ")
println(ksol)
println("Krylov Norm:", norm(kdif))
println()

include("myownshit.jl")
xs, = mybicgstab(A, b)
x = view(xs, :, size(xs, 2))

println("My stuff")
println("solution:")
println(x)

dif = b - A * x
println(norm(dif))

solch = map(x -> norm(b - A * x), eachcol(xs))
println("Solution change:", solch)
println("with minimum:", minimum(solch))