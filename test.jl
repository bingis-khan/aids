using IncompleteLU
import IterativeSolvers
import Krylov

# test how much two matrices(vectors) differ
matrixdistance(l, r) = sqrt(sum((l .- r) .^ 2)) 


function testboth(A, b)
    # precondition
    Ap = ilu(A)
    
    # testing iterative solvers
    

    # Testing krylov
end
