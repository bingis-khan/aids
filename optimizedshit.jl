using LinearAlgebra
using SparseArrays


function isaccenough(A, x, b)
    b - A * x
end

# index by column
@inline function c(v, i)
    view(v, :, i)
end


function newbicgstab(A::SparseMatrixCSC{Float64, Int64}, b::Vector{Float64}, Pl)
    # initialize all of this shiet
    # less for performance, more for debugging
    # remember each step
    T = eltype(b)
    n, nump = size(A)
    max_it = 2 * n
    x = zeros(T, nump, max_it+1)
    r = Matrix{T}(undef, n, max_it+1)
    copyto!(view(r, :, 1), b)

    r_hat = c(r, 1)

    @assert(r_hat == b)

    # scalars
    rho = ones(max_it+1)
    omega = copy(rho)
    alpha::Float64 = 1

    # vectors 
    v = zeros(T, nump, max_it + 1) 
    p = copy(v)

    tol = sqrt(eps(T))

    # statically allocate some memory
    # only y has some speedup for some reason, the lower amount of allocations, but increase running time
    y = Vector{T}(undef, nump)    

    @inbounds for i in 2:max_it + 1
        rho[i] = dot(r_hat, c(r, i-1))
        beta = (rho[i] / rho[i-1]) * (alpha / omega[i-1])
        # println(beta)
        copyto!(c(p, i), c(r, i-1) + beta * (c(p, i-1) - omega[i-1] * c(v, i-1)))
        ldiv!(y, Pl, c(p, i))
        copyto!(c(v, i), A*y)
        alpha = rho[i] / dot(r_hat, c(v, i))
        h = c(x, i-1) + alpha * y

        # if h is accurate enough...
        if norm(b - A * h) <= tol
            copyto!(c(x, i), h)
            return c(x, i), i - 1, x
        end

        s = c(r, i-1) - alpha * c(v, i)
        z = Pl \ s
        t = A * z
        plt = Pl \ t
        pls =  Pl \ s
        omega[i] = dot(plt, pls) / dot(plt, plt)
        copyto!(c(x, i), h + omega[i] * z)

        if norm(b - A * c(x, i)) <= tol
            return c(x, i), i - 1, x
        end

        copyto!(c(r, i), s - omega[i] * t)
    end

    c(x, max_it + 1), max_it, x
end
