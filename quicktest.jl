
include("generators/symmetric-problem-fem.jl")


A, b = femproblem(20)



using Krylov
using LinearAlgebra
using IncompleteLU
using SparseArrays
using LimitedLDLFactorizations
using Preconditioners
using AlgebraicMultigrid

function incomplete_cholesky_preconditioner(A::SparseMatrixCSC{T}) where T<:Real
    a = A
    n = size(a,1);

    	for k = 1:n
    		a[k,k] = sqrt(a[k,k]);
    		for i = (k+1):n
    		    if (a[i,k] != 0)
    		        a[i,k] = a[i,k]/a[k,k];            
    		    end
    		end
    		for j = (k+1):n
    		    for i = j:n
    		        if a[i,j] != 0
    		            a[i,j] = a[i,j] - a[i,k]*a[j,k]  
    		        end
    		    end
    		end
    	end

        for i in 1:n
            for j in i+1:n
                a[i,j] = 0
            end
        end
    n
end


# Pl = DiagonalPreconditioner(A)
# Pl = incomplete_cholesky_preconditioner(A)
# Pl = AMGPreconditioner{SmoothedAggregation}(A)
Pl = aspreconditioner(ruge_stuben(A))
# Pl = I
# Pl = ilu(A)
# Pl = lldl(A)
# Pl.D .=  abs.(Pl.D)
ksol, stats = bicgstab(A, b, M=Pl, ldiv=true)

kdif = b - A * ksol
# println("solution: ")
# println(ksol)
println("Krylov Norm:", norm(kdif))
println(stats)
println()

include("myownshit.jl")
x, i, xs = mybicgstab(A, b, Pl)

xs = view(xs, :, 1:i+1)

println("My stuff")
# println("solution:")
# println(x)

dif = b - A * x
println(norm(dif))

solch = map(x -> norm(b - A * x), eachcol(xs))
# println("Solution change:", solch)
println("with minimum:", minimum(solch))
println("Iterations:", i)