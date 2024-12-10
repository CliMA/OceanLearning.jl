using Documenter, Test

# include("test_lesbrary_datadeps.jl")
include("test_synthetic_observations.jl")
include("test_transformations.jl")
include("test_parameters.jl")
include("test_forward_map.jl")
include("test_lesbrary_forward_map.jl")
include("test_ensemble_models.jl")
include("test_ensemble_kalman_inversion.jl")
include("test_utils.jl")

@testset "Doctests" begin
    using ParameterEstimocean, Distributions
    doctest(ParameterEstimocean)
end
