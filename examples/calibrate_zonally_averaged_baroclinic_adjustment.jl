# Calibration of Gent-McWilliams to a baroclinic adjustment problem
pushfirst!(LOAD_PATH, joinpath(@__DIR__, ".."))

using ParameterEstimocean
using Oceananigans
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: SliceEnsembleSize
using Oceananigans.TurbulenceClosures: FluxTapering
using LinearAlgebra, CairoMakie, DataDeps, JLD2

using Oceananigans.Architectures: on_architecture

# using ElectronDisplay

architecture = CPU()

# filedir = @__DIR__
# filename = "baroclinic_adjustment_double_Lx_zonal_average.jld2"
# filepath = joinpath(filedir, filename)
# Base.download("https://www.dropbox.com/s/f8zsb33vwwwmjjm/$filename", filepath)

# filepath = "/Users/navid/Research/mesoscale-parametrization-OSM2022/baroclinic_adjustment-double_Lx/short_save_often_run/baroclinic_adjustment_double_Lx_zonal_average.jld2"
filepath = "baroclinic_adjustment_double_Lx_zonal_average_80dayrun.jld2"

file = jldopen(filepath)
coriolis = file["serialized/coriolis"]

# number of grid points
Nx, Ny, Nz = file["grid/Nx"], file["grid/Ny"], file["grid/Nz"]

# Domain
const Lx, Ly, Lz = file["grid/Lx"], file["grid/Ly"], file["grid/Lz"]

close(file)


field_names = (:b, :c, :u, :v)
forward_map_names = (:b, :c)

using ParameterEstimocean.Transformations: Transformation

transformation = (b = ZScore(),
                  c = ZScore(),
                  u = ZScore())

transformation = ZScore()

space_transformation = SpaceIndices(x=:, y=2:4:Ny-1, z=20:2:Nz-1)

transformation = (b = Transformation(space = space_transformation, normalization=ZScore()),
                  c = Transformation(space = space_transformation, normalization=ZScore()),
                  u = Transformation(space = space_transformation, normalization=ZScore()),
                  v = Transformation(space = space_transformation, normalization=RescaledZScore(1e-1)))

transformation = ZScore()

transformation = (b = Transformation(space = space_transformation, normalization=ZScore()),
                  c = Transformation(space = space_transformation, normalization=ZScore()))

times = [40days-12hours, 40days]

observations = SyntheticObservations(filepath; transformation, times, field_names, forward_map_names)

#####
##### Simulation
#####

Nensemble = 50

slice_ensemble_size = SliceEnsembleSize(size=(Ny, Nz), ensemble=Nensemble)

ensemble_grid = RectilinearGrid(architecture,
                                size = slice_ensemble_size,
                                topology = (Flat, Bounded, Bounded),
                                y = (-Ly/2, Ly/2),
                                z = (-Lz, 0),
                                halo=(3, 3))

@show ensemble_grid

file = jldopen(filepath)
closures = file["serialized/closure"]
close(file)

gent_mcwilliams_diffusivity = IsopycnalSkewSymmetricDiffusivity(slope_limiter = FluxTapering(1e-2))

gm_closure_ensemble = on_architecture(architecture, [deepcopy(gent_mcwilliams_diffusivity) for _ = 1:Nensemble])

closure_ensemble = (gm_closure_ensemble, closures[1], closures[2])

@show "no convective adjustment"
closure_ensemble = (gm_closure_ensemble, closures[1])

ensemble_model = HydrostaticFreeSurfaceModel(grid = ensemble_grid,
                                             tracers = (:b, :c),
                                             buoyancy = BuoyancyTracer(),
                                             coriolis = coriolis,
                                             closure = closure_ensemble,
                                             free_surface = ImplicitFreeSurface())

Δt = 1.0
simulation = Simulation(ensemble_model; Δt, stop_time=times[end])

priors = (
     κ_skew = ScaledLogitNormal(bounds = (300, 3000)),
     κ_symmetric = ScaledLogitNormal(bounds = (300, 3000))
 )

free_parameters = FreeParameters(priors)

using ParameterEstimocean.Observations: FieldTimeSeriesCollector

simulation_fields = fields(simulation.model)
collected_fields = NamedTuple(name => simulation_fields[name] for name in ParameterEstimocean.Observations.forward_map_names(observations))
time_series_collector = FieldTimeSeriesCollector(collected_fields, observation_times(observations), architecture=CPU())

calibration = InverseProblem(observations, simulation, free_parameters; time_series_collector)

eki = EnsembleKalmanInversion(calibration;
                              noise_covariance = 5e-3,
                              resampler = Resampler(acceptable_failure_fraction=1.0))

iterate!(eki; iterations = 10)


# Last, we visualize few metrics regarding how the EKI calibration went about.

y = observation_map(calibration)
#=
θ̅(iteration) = [eki.iteration_summaries[iteration].ensemble_mean...]
varθ(iteration) = eki.iteration_summaries[iteration].ensemble_var

weight_distances = [norm(θ̅(iter)) for iter in 1:eki.iteration]
output_distances = [norm(forward_map(calibration, θ̅(iter))[:, 1] - y) for iter in 1:eki.iteration]
ensemble_variances = [varθ(iter) for iter in 1:eki.iteration]


f = Figure()
lines(f[1, 1], 1:eki.iteration, weight_distances, color = :red, linewidth = 2,
      axis = (title = "Parameter norm",
              xlabel = "Iteration",
              ylabel="|θ̅ₙ|",
              yscale = log10))
lines(f[1, 2], 1:eki.iteration, output_distances, color = :blue, linewidth = 2,
      axis = (title = "Output distance",
              xlabel = "Iteration",
              ylabel="|G(θ̅ₙ) - y|",
              yscale = log10))
ax3 = Axis(f[2, 1:2], title = "Parameter convergence",
           xlabel = "Iteration",
           ylabel = "Ensemble variance",
           yscale = log10)

for (i, pname) in enumerate(free_parameters.names)
    ev = getindex.(ensemble_variances, i)
    lines!(ax3, 1:eki.iteration, ev / ev[1], label = String(pname), linewidth = 2)
end

axislegend(ax3, valign = :top, halign = :right)

save("summary_bca.png", f); nothing #hide 
save("summary_bca.svg", f); nothing #hide 

# ![](summary_channel.svg)

# And also we plot the the distributions of the various model ensembles for few EKI iterations to see
# if and how well they converge to the true diffusivity values.

=#

f = Figure()

axtop = Axis(f[1, 1])

axmain = Axis(f[2, 1],
              xlabel = "κ_skew [m² s⁻¹]",
              ylabel = "κ_symmetric [m² s⁻¹]")

axright = Axis(f[2, 2])
scatters = []
labels = String[]

for iter in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ## Make parameter matrix
    parameters = eki.iteration_summaries[iter].parameters
    Nensemble = length(parameters)
    Nparameters = length(first(parameters))
    parameter_ensemble_matrix = [parameters[i][j] for i=1:Nensemble, j=1:Nparameters]

    label = iter == 0 ? "Initial ensemble" : "Iteration $iter"
    push!(labels, label)
    push!(scatters, scatter!(axmain, parameter_ensemble_matrix))
    density!(axtop, parameter_ensemble_matrix[:, 1])
    density!(axright, parameter_ensemble_matrix[:, 2], direction = :y)
end

colsize!(f.layout, 1, Fixed(300))
colsize!(f.layout, 2, Fixed(200))

rowsize!(f.layout, 1, Fixed(200))
rowsize!(f.layout, 2, Fixed(300))

Legend(f[1, 2], scatters, labels, valign = :bottom, halign = :left)

hidedecorations!(axtop, grid = false)
hidedecorations!(axright, grid = false)

xlims!(axright, 0, 0.025)
ylims!(axtop, 0, 0.025)

save("distributions_bca_2.png", f); nothing #hide 
save("distributions_bca_2.svg", f); nothing #hide 

# ![](distributions_channel.svg)
