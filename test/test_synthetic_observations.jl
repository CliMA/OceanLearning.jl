using Test
using DataDeps
using ParameterEstimocean
using Oceananigans
using Oceananigans.Units
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ColumnEnsembleSize
using Oceananigans.TurbulenceClosures: ConvectiveAdjustmentVerticalDiffusivity

@testset "SyntheticObservations tests" begin
    @info "  Generating synthetic observations with an Oceananigans.Simulation..."
    # Generate synthetic observations
    Nz = 16
    Lz = 128
    Jᵇ = 1e-8
    Jᵘ = -1e-5
    Δt = 20.0
    f₀ = 1e-4
    N² = 1e-5
    
    stop_time = 6Δt
    save_interval = 2Δt
    N_ensemble = 3
    
    experiment_name = "convective_adjustment_test"
    data_path = experiment_name * ".jld2"
    
    # "True" parameters to be estimated by calibration
    convective_κz = 1.0
    convective_νz = 0.9
    background_κz = 1e-4
    background_νz = 1e-5

    grid = RectilinearGrid(size=Nz, z=(-Lz, 0), topology=(Flat, Flat, Bounded))
    closure = ConvectiveAdjustmentVerticalDiffusivity(; convective_κz, background_κz, convective_νz, background_νz)
    coriolis = FPlane(f=f₀)

    u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Jᵘ))
    b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Jᵇ), bottom = GradientBoundaryCondition(N²))

    model = HydrostaticFreeSurfaceModel(; grid, coriolis, closure,
                                        tracers = :b,
                                        buoyancy = BuoyancyTracer(),
                                        boundary_conditions = (; u=u_bcs, b=b_bcs))

    set!(model, b = z -> N² * z)
    simulation = Simulation(model; Δt, stop_time)
    init_with_parameters(file, model) = file["parameters"] = (; Jᵇ, Jᵘ, Δt, N², tracers=keys(model.tracers))
    simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                                                          schedule = TimeInterval(save_interval),
                                                          filename = experiment_name,
                                                          with_halos = true,
                                                          overwrite_existing = true,
                                                          init = init_with_parameters)

    run!(simulation)

    #####
    ##### Test synthetic observations construction
    #####
    
    @testset "SyntheticObservations construction" begin
        @info "    Testing construction of SyntheticObservations from Oceananigans data..."
        data_path = experiment_name * ".jld2"

        # field_names
        b_observations = SyntheticObservations(data_path, field_names=:b)
        ub_observations = SyntheticObservations(data_path, field_names=(:u, :b))
        uvb_observations = SyntheticObservations(data_path, field_names=(:u, :v, :b))

        @test keys(b_observations.field_time_serieses) == tuple(:b)
        @test keys(ub_observations.field_time_serieses) == tuple(:u, :b)
        @test keys(uvb_observations.field_time_serieses) == tuple(:u, :v, :b)

        # Batched observations
        batch = BatchedSyntheticObservations((b_observations, ub_observations))
        @test batch isa BatchedSyntheticObservations
        @test batch[1] isa SyntheticObservations
        @test length(batch.observations) == 2
        @test batch.weights == (1, 1)

        # transformations and normalizations
        field_names = (:u, :v, :b)
        transformation = ZScore()
        uvb_observations = SyntheticObservations(data_path; field_names, transformation)
        @test all(t.normalization isa ZScore for t in values(uvb_observations.transformation))

        transformation = (u = ZScore(), v = ZScore(), b = ZScore())
        uvb_observations = SyntheticObservations(data_path; field_names, transformation)
        @test all(t.normalization isa ZScore for t in values(uvb_observations.transformation))

        transformation = (u = nothing, v = ZScore(), b = RescaledZScore(0.1))
        uvb_observations = SyntheticObservations(data_path; field_names, transformation)
        @test uvb_observations.transformation[:u].normalization isa Nothing
        @test uvb_observations.transformation[:v].normalization isa ZScore
        @test uvb_observations.transformation[:b].normalization isa RescaledZScore
        @test uvb_observations.transformation[:b].normalization.scale === 0.1

        # Regridding
        coarse_grid = RectilinearGrid(size=Int(Nz/2), z=(-Lz, 0), topology=(Flat, Flat, Bounded))
        fine_grid = RectilinearGrid(size=2Nz, z=(-Lz, 0), topology=(Flat, Flat, Bounded))

        for regrid in [(1, 1, Int(Nz/2)), coarse_grid]
            coarsened_observations = SyntheticObservations(data_path, field_names=(:u, :v, :b); regrid)
            @test size(coarsened_observations.grid) === (1, 1, Int(Nz/2))
        end

        for regrid in [(1, 1, 2Nz), fine_grid]
            refined_observations = SyntheticObservations(data_path, field_names=(:u, :v, :b); regrid)
            @test size(refined_observations.grid) === (1, 1, 2Nz)
        end

        # Test regridding LESbrary observations
        data_path = datadep"two_day_suite_2m/free_convection_instantaneous_statistics.jld2";
        for Nz in (8, 16, 32, 64, 128, 256, 512)
            observations = SyntheticObservations(data_path; field_names=(:u, :v, :b), regrid=(1, 1, Nz))
            @test size(observations.grid) === (1, 1, Nz)

            raw_obs = SyntheticObservations(data_path; field_names=(:u, :v, :b))
            regrid = RectilinearGrid(size=Nz, z=(-raw_obs.grid.Lz, 0), topology=(Flat, Flat, Bounded))
            observations = SyntheticObservations(data_path; field_names=(:u, :v, :b), regrid)
            @test size(observations.grid) === (1, 1, Nz)
        end
    end

    @testset "SyntheticObservations display" begin
        data_path = experiment_name * ".jld2"
        observations = SyntheticObservations(data_path, field_names=(:u, :v, :b))
        @show observations 
        @test true
    end
end
