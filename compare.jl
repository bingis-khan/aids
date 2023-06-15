using BenchmarkTools
using Plots
using IncompleteLU

# generators
include("generators/symmetric-problem-fdm.jl")
include("generators/symmetric-problem-fem.jl")
include("generators/nonsymmetric-problem-fvm.jl")

# for n: 1, 2, 3..., generate for example 1000 problems and test on them, then average times. Also, for every type of problem.
# krylov sets maxiter as 2 * size of matrix. IterativeSolvers has just 2. I'll take krylov - seems more legit. 
# nvm, IS bicgstabl has a tendency to Inf/NaN the matrix when given 2*size
# krylov is also faster and has better accuracy...

matrixdistance(l, r) = sqrt(sum((l .- r) .^ 2)) 

# the matrix distance is similar, so whatever

testone(tf, gf) = @benchmark $tf(data[1], data[2]) setup=(data=$gf()) samples=5
todatapoint(t::BenchmarkTools.Trial) = (time(median(t)), time(mean(t)))


function precon(gf)
    A, s = gf()
    Alu = ilu(A)
    Alu, s
end

function bench(range, tf, gf)
    meds::Array{Float64}, avgs::Array{Float64} = [], []
    for d in range
        med, avg = todatapoint(testone(tf, () -> precon(gf(d))))
        push!(meds, med)
        push!(avgs, avg)
    end

    meds, avgs
end

benchallproblems(range, tf) = (bench(range, tf, femproblem), bench(range, tf, fdmproblem), bench(range, tf, fvmproblem))


import Krylov
import IterativeSolvers



preparethrumagick(xs) = [[b for b in zip(a...)] for a in zip(xs...)] 

function plotcomparison()
    range = 4:6
    it = benchallproblems(range, (a, s) -> IterativeSolvers.bicgstabl(a, s, log = false))
    krylov = benchallproblems(range, Krylov.bicgstab)
    tr = preparethrumagick((it, krylov))
    fem, fdm, fvm = tr[1], tr[2], tr[3]

    femavgsit, femavgskr = fem[1]
    @show femavgsit
    plot(range, femavgsit)
    plot!(range, femavgskr)
    
end