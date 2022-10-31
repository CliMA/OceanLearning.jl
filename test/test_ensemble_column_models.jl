using Test
using Oceananigans

using Oceananigans.Models.HydrostaticFreeSurfaceModels: ColumnEnsembleSize, SliceEnsembleSize
using Oceananigans.TurbulenceClosures: ConvectiveAdjustmentVerticalDiffusivity

const CAVD = ConvectiveAdjustmentVerticalDiffusivity

@testset "Ensembles of `HydrostaticFreeSurfaceModel` with different closures" begin

    Nz = 16
    Hz = 1
    topology = (Flat, Flat, Bounded)
    grid = RectilinearGrid(; size=Nz, z=(-10, 10), topology, halo=Hz)

    closures = [CAVD(background_κz=1.0) CAVD(background_κz=1.1)
                CAVD(background_κz=1.2) CAVD(background_κz=1.3)
                CAVD(background_κz=1.4) CAVD(background_κz=1.5)]

    ensemble_size = size(closures)

    @test size(closures) == (3, 2)
    @test closures[2, 1].background_κz == 1.2 

    Δt = 0.01 * grid.Δzᵃᵃᶜ^2

    model_kwargs = (; tracers=:c, buoyancy=nothing, coriolis=nothing)
    simulation_kwargs = (; Δt, stop_iteration=100)

    models = [HydrostaticFreeSurfaceModel(; grid, closure=closures[i, j], model_kwargs...) for i=1:ensemble_size[1], j=1:ensemble_size[2]]

    set_ic!(model) = set!(model, c = (x, y, z) -> exp(-z^2)) 

    for model in models
        set_ic!(model)
        simulation = Simulation(model; simulation_kwargs...)
        run!(simulation)
    end 

    ensemble_grid = RectilinearGrid(; size=ColumnEnsembleSize(; Nz, ensemble=ensemble_size, Hz),
                                      z=(-10, 10), topology, halo=Hz)

    @test size(ensemble_grid) == (ensemble_size[1], ensemble_size[2], Nz)

    ensemble_model = HydrostaticFreeSurfaceModel(; grid=ensemble_grid, closure=closures, model_kwargs...)
    set_ic!(ensemble_model)

    @test size(parent(ensemble_model.tracers.c)) == (ensemble_size[1], ensemble_size[2], Nz+2)

    ensemble_simulation = Simulation(ensemble_model; simulation_kwargs...)
    run!(ensemble_simulation)

    for i=1:ensemble_size[1], j=1:ensemble_size[2]
        @info "Testing ConvectiveAdjustmentVerticalDiffusivity ensemble member ($i, $j)..."
        @test parent(ensemble_model.tracers.c)[i, j, :] == parent(models[i, j].tracers.c)[1, 1, :]
    end

end

@testset "Ensembles of column `HydrostaticFreeSurfaceModel`s with different Coriolis parameters" begin

    Nz = 2
    Hz = 1
    topology = (Flat, Flat, Bounded)

    grid = RectilinearGrid(; size=Nz, z=(-1, 0), topology, halo=Hz)

    coriolises() = [FPlane(f=1.0) FPlane(f=1.1) FPlane(f=2.1)
                    FPlane(f=1.0) FPlane(f=1.1) FPlane(f=2.1)]

    ensemble_size = size(coriolises())

    Δt = 0.01

    @test size(coriolises()) == (2, 3)
    @test coriolises()[2, 2].f == 1.1

    model_kwargs = (; tracers=nothing, buoyancy=nothing, closure=nothing)
    simulation_kwargs = (; Δt, stop_time=2π)

    models = [HydrostaticFreeSurfaceModel(; grid, coriolis=coriolises()[i, j], model_kwargs...) for i=1:ensemble_size[1], j=1:ensemble_size[2]]

    set_ic!(model) = set!(model, u=sqrt(2), v=sqrt(2))

    for model in models
        set_ic!(model)
        simulation = Simulation(model; simulation_kwargs...)
        run!(simulation)
    end 

    ensemble_grid = RectilinearGrid(; size=ColumnEnsembleSize(; Nz, ensemble=ensemble_size, Hz),
                                      z=(-1, 0), topology, halo=Hz)
    ensemble_model = HydrostaticFreeSurfaceModel(; grid=ensemble_grid, coriolis=coriolises(), model_kwargs...)
    set_ic!(ensemble_model)
    ensemble_simulation = Simulation(ensemble_model; simulation_kwargs...)
    run!(ensemble_simulation)

    for i=1:ensemble_size[1], j=1:ensemble_size[2]
        @info "Testing Coriolis ensemble member ($i, $j) with $(coriolises()[i, j])..."
        @test ensemble_model.coriolis[i, j] == coriolises()[i, j]

        # @show parent(ensemble_model.velocities.u)[i, j, :]
        # @show parent(models[i, j].velocities.u)[1, 1, :]

        u = parent(ensemble_model.velocities.u)[i, 1, :]
        v = parent(ensemble_model.velocities.v)[i, 1, :]
        @show @. u^2 + v^2

        u = parent(models[i, j].velocities.u)[i, 1, :]
        v = parent(models[i, j].velocities.v)[i, 1, :]
        @show @. u^2 + v^2

        # @test parent(ensemble_model.velocities.u)[i, j, :] == parent(models[i, j].velocities.u)[1, 1, :]

        # @show parent(ensemble_model.velocities.v)[i, j, :]
        # @show parent(models[i, j].velocities.v)[1, 1, :]
        # @test parent(ensemble_model.velocities.v)[i, j, :] == parent(models[i, j].velocities.v)[1, 1, :]
    end

end

#=
@testset "Ensembles of slice `HydrostaticFreeSurfaceModel`s with different Coriolis parameters" begin

    Ny, Nz = 4, 2
    Hy, Hz = 1, 1
    grid = RectilinearGrid(size=(Ny, Nz), y = (-10, 10), z=(-1, 0), topology=(Flat, Bounded, Bounded), halo=(Hy, Hz))

    coriolises = [FPlane(f=1.0), FPlane(f=1.1), FPlane(f=1.2)]

    Nensemble = length(coriolises)

    Δt = 0.01

    @test size(coriolises) == (3,)
    @test coriolises[2].f == 1.1

    model_kwargs = (; tracers=nothing, buoyancy=nothing, closure=nothing)
    simulation_kwargs = (; Δt, stop_iteration=100)

    models = [HydrostaticFreeSurfaceModel(; grid, coriolis=coriolises[i], model_kwargs...) for i=1:Nensemble]

    set_ic!(model) = set!(model, u=sqrt(2), v=sqrt(2))

    for model in models
        set_ic!(model)
        simulation = Simulation(model; simulation_kwargs...)
        run!(simulation)
    end 

    ensemble_size = SliceEnsembleSize(size=(Ny, Nz), ensemble=Nensemble)
    ensemble_grid = RectilinearGrid(size=ensemble_size, y = (-10, 10), z=(-1, 0), topology=(Flat, Bounded, Bounded), halo=(Hy, Hz))
    ensemble_model = HydrostaticFreeSurfaceModel(; grid=ensemble_grid, coriolis=coriolises, model_kwargs...)
    set_ic!(ensemble_model)
    ensemble_simulation = Simulation(ensemble_model; simulation_kwargs...)
    run!(ensemble_simulation)

    for i = 1:Nensemble
        @info "Testing Coriolis ensemble member ($i,) with $(coriolises[i])..."
        @test ensemble_model.coriolis[i] == coriolises[i]

        # @show parent(ensemble_model.velocities.u)[i, 1, :]
        # @test parent(ensemble_model.velocities.u)[i, 1, :] == parent(models[i].velocities.u)[1, 1, :]

        # @show parent(ensemble_model.velocities.v)[i, 1, :]
        # @test parent(ensemble_model.velocities.v)[i, 1, :] == parent(models[i].velocities.v)[1, 1, :]
    end

end
=#