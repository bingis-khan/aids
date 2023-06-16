
include("generators/symmetric-problem-fdm.jl")


A, b = fdmproblem(50)

using Krylov
using LinearAlgebra
using IncompleteLU


LU = ilu(A)
ksol, = bicgstab(A, b, M=LU, ldiv=true)

kdif = b - A * ksol
println("solution: ")
println(ksol)
println("Krylov Norm:", norm(kdif))
println()

include("myownshit.jl")
x, i, xs = mybicgstab(A, b)

xs = view(xs, :, 1:i+1)

println("My stuff")
println("solution:")
println(x)

dif = b - A * x
println(norm(dif))

solch = map(x -> norm(b - A * x), eachcol(xs))
println("Solution change:", solch)
println("with minimum:", minimum(solch))