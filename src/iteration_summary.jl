using ..Parameters: transform_to_unconstrained, build_parameters_named_tuple

using Oceananigans.Utils: prettysummary

struct IterationSummary{U, P, M, C, V, E, O}
    unconstrained_parameters :: U
    parameters :: P     # constrained
    ensemble_mean :: M  # constrained
    ensemble_cov :: C   # constrained
    ensemble_var :: V
    mean_square_errors :: E
    objective_values :: O
    iteration :: Int
    pseudotime :: Float64
    pseudo_Δt :: Float64
end

"""
    eki_objective(eki, θ::AbstractVector, G::AbstractVector; constrained = false)

Given forward map `G` and parameters `θ`, return a tuple `(Φ₁, Φ₂)` 
of terms in the ensemble Kalman inversion (`eki`) regularized objective
function, where,

```code
Φ = Φ₁ + Φ₂
```

`Φ₁` measures output misfit `½ || Γy^(-¹/₂) * (y .- G(θ)) ||²` and 
`Φ₂` measures prior misfit `½ || Γθ^(-¹/₂) * (θ .- μθ) ||²`, where `y` is the observation 
map, `G(θ)` is the forward map, `Γy` is the observation noise covariance, `Γθ` is 
the prior covariance, and `μθ` represents the prior means. Note that `Γ^(-¹/₂) = 
inv(sqrt(Γ))`.

When keyword argument `constrained` is provided with `true` then input `θ`
is assumed to represent constrained parameters.
"""
function eki_objective(eki, θ, G::AbstractVector; constrained=false, augmented=false)
    y = eki.mapped_observations
    Γy = eki.noise_covariance
    inv_sqrt_Γy = eki.precomputed_arrays[:inv_sqrt_Γy]
    inv_sqrt_Γθ = eki.precomputed_arrays[:inv_sqrt_Γθ]
    μθ = eki.precomputed_arrays[:μθ]

    if constrained
        priors = eki.inverse_problem.free_parameters.priors
        θ = [transform_to_unconstrained(priors[name], θ[i]) for (i, name) in enumerate(keys(priors))]
    end

    if augmented
        y = eki.precomputed_arrays[:y_augmented]
        inv_sqrt_Σ = eki.precomputed_arrays[:inv_sqrt_Σ]
        η̃ = eki.precomputed_arrays[:η_mean_augmented]
        ϵ = inv_sqrt_Σ * (y - G - η̃)
        Φ₁ = norm(ϵ)^2 / 2
        return (Φ₁, 0)
    end

    # Φ₁ = 1/2 * || Γy^(-½) * (y - G) ||²
    ϵ = inv_sqrt_Γy * (y .- G)
    Φ₁ = norm(ϵ)^2 / 2

    # Φ₂ = 1/2 * || Γθ^(-½) * (θ - μθ) ||² 
    if eki.tikhonov
        χ = inv_sqrt_Γθ * (θ .- μθ)
        Φ₂ = norm(χ)^2 / 2
    else
        Φ₂ = 0
    end

    return (Φ₁, Φ₂)
end

"""
    IterationSummary(eki, X, forward_map_output=nothing)

Return the summary for ensemble Kalman inversion `eki` 
with unconstrained parameters `X` and `forward_map_output`.
"""
function IterationSummary(eki, X, forward_map_output=nothing)
    priors = eki.inverse_problem.free_parameters.priors

    ensemble_mean = mean(X, dims=2)[:] 
    constrained_ensemble_mean = transform_to_constrained(priors, ensemble_mean)

    ensemble_covariance = cov(X, dims=2)
    constrained_ensemble_covariance = inverse_covariance_transform(values(priors), X, ensemble_covariance)

    constrained_ensemble_variance = build_parameters_named_tuple(eki.inverse_problem.free_parameters,
                                                                 diag(constrained_ensemble_covariance),
                                                                 with_dependent_parameters = false)

    constrained_parameters = transform_to_constrained(priors, X)

    G = forward_map_output
    if !isnothing(forward_map_output)
        Nobs, Nens= size(forward_map_output)
        y = eki.mapped_observations
        mean_square_errors = [mapreduce((x, y) -> (x - y)^2, +, y, view(G, :, k)) / Nobs for k = 1:Nens]
    else
        mean_square_errors = nothing
    end

    # Vector of (Φ₁, Φ₂) pairs, one for each ensemble member at the current iteration
    objective_values = [eki_objective(eki, X[:, j], G[:, j]) for j in 1:size(G, 2)]

    return IterationSummary(X,
                            constrained_parameters,
                            constrained_ensemble_mean,
                            constrained_ensemble_covariance,
                            constrained_ensemble_variance,
                            mean_square_errors,
                            objective_values,
                            eki.iteration,
                            eki.pseudotime,
                            eki.pseudo_Δt)
end

function finitefind(f, a, val, find)
    fa = f.(a)
    finite_fa = f.(a)
    finite_fa[.!isfinite.(fa)] .= val
    return find(finite_fa)
end

finitefindmin(a) = finitefind(identity, a, Inf, findmin)
finitefindmax(a) = finitefind(identity, a, -Inf, findmax)
finitefindmin(f, a) = finitefind(f, a, Inf, findmin)
finitefindmax(f, a) = finitefind(f, a, -Inf, findmax)

function Base.show(io::IO, is::IterationSummary)
    max_error, imax = finitefindmax(o -> o[1] + o[2], is.objective_values)
    min_error, imin = finitefindmin(o -> o[1] + o[2], is.objective_values)

    names = keys(is.ensemble_mean)
    parameter_matrix = [is.parameters[k][name] for name in names, k = 1:length(is.parameters)]
    min_parameters = minimum(parameter_matrix, dims=2)
    max_parameters = maximum(parameter_matrix, dims=2)

    print(io, summary(is), '\n')

    print(io, "                      ", param_str.(keys(is.ensemble_mean))..., '\n',
              "       ensemble_mean: ", param_str.(values(is.ensemble_mean))..., '\n',
              particle_str("best",  sum(is.objective_values[imin]), is.parameters[imin]), '\n',
              particle_str("worst", sum(is.objective_values[imax]), is.parameters[imax]), '\n',
              "             minimum: ", param_str.(min_parameters)..., '\n',
              "             maximum: ", param_str.(max_parameters)..., '\n',
              "   ensemble_variance: ", param_str.(values(is.ensemble_var))...)

    return nothing
end

Base.summary(is::IterationSummary) = string("IterationSummary(",
                                            "iteration=", is.iteration,
                                            ", pseudotime=", prettysummary(is.pseudotime),
                                            ", pseudo_Δt=", prettysummary(is.pseudo_Δt), ") ",
                                            "for ", length(is.parameters),
                                            " particles and ", length(keys(is.ensemble_mean)),
                                            " parameters")

function param_str(p::Symbol)
    p_str = string(p)
    length(p_str) > 9 && (p_str = p_str[1:9])
    return @sprintf("% 10s | ", p_str)
end

param_str(p::Number) = @sprintf("% -1.3e | ", p)

particle_str(particle, ΣΦ, parameters) =
    @sprintf("% 11s particle: ", particle) *
    string(param_str.(values(parameters))...) *
    @sprintf("objective = %.6e", ΣΦ)

