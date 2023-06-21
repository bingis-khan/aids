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

    x = zeros(T, nump)
    px = zeros(T, nump)

    r = Vector{T}(undef, n)
    pr = Vector{T}(undef, n)
    copyto!(r, b)
    copyto!(pr, b)

    r_hat = r

    @assert(r_hat == b)

    # scalars
    rho = prho = 1
    omega = pomega = 1
    alpha::Float64 = 1

    # vectors 
    v = zeros(nump)
    pv = zeros(nump)
    p = zeros(nump)
    pp = zeros(nump)

    tol = sqrt(eps(T))

    @inbounds for i in 2:max_it + 1
        rho = dot(r_hat, pr)
        beta = (rho / prho) .* (alpha / pomega)
        p = pr .+ beta .* (pp - pomega .* pv)
        y = Pl \ p
        v = A*y
        alpha = rho / dot(r_hat, v)
        h = px .+ alpha .* y


        # if h is accurate enough...
        if norm(b - A * h) <= tol
            return h, i - 1
        end

        s = pr - alpha .* v
        z = Pl \ s
        t = A * z
        plt = Pl \ t
        pls =  Pl \ s
        omega = dot(plt, pls) / dot(plt, plt)
        x = h + omega .* z

        if norm(b - A * x) <= tol
            return x, i - 1
        end

        r = s - omega .* t

        px = x
        pr = r
        pp = p
        pv = v
        prho = rho
        pomega = omega
    end

    x, max_it
end
