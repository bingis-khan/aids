using LinearAlgebra
using IncompleteLU


function isaccenough(A, x, b)
    b - A * x
end

# index by column
function c(v, i)
    view(v, :, i)
end


function mybicgstab(A, b)
    # preconditioning
    Pl = ilu(A) 

    # initialize all of this shiet
    # less for performance, more for debugging
    # remember each step
    T = eltype(b)
    n = size(A, 1)
    max_it = 20 * n
    x = zeros(T, size(A, 2), max_it+1)
    r = Matrix{T}(undef, n, max_it+1)
    copyto!(view(r, :, 1), b)

    r_hat = c(r, 1)

    @assert(r_hat == b)

    # scalars
    rho = ones(max_it+1)
    omega = copy(rho)
    alpha = 1

    # vectors 
    v = zeros(T, size(A, 2), max_it + 1) 
    p = copy(v)

    tol = sqrt(eps(T))

    for i in 2:max_it + 1
        goodenough(x) = norm(b - A * x) <= tol

        rho[i] = dot(r_hat, c(r, i-1))
        beta = (rho[i] / rho[i-1]) * (alpha / omega[i-1])
        # println(beta)
        copyto!(c(p, i), c(r, i-1) + beta * (c(p, i-1) - omega[i-1] * c(v, i-1)))
        y = Pl \ c(p, i)
        copyto!(c(v, i), A*y)
        alpha = rho[i] / dot(r_hat, c(v, i))
        h = c(x, i-1) + alpha * y

        # if h is accurate enough...
        if goodenough(h)
            copyto!(c(x, i), h)
            return h, i - 1, x, r, p, v, rho, omega
        end

        s = c(r, i-1) - alpha * c(v, i)
        z = Pl \ s
        t = A * z
        plt = Pl \ t
        omega[i] = dot(plt, Pl \ s) / dot(plt, plt)
        copyto!(c(x, i), h + omega[i] * z)

        if goodenough(c(x, i))
            return c(x, i), i - 1, x, r, p, v, rho, omega
        end

        copyto!(c(r, i), s - omega[i] * t)
    end

    c(x, max_it + 1), max_it, x, r, p, v, rho, omega
end
