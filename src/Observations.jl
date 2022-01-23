module Observations

using Oceananigans
using Oceananigans: fields
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Grids: cpu_face_constructor_x, cpu_face_constructor_y, cpu_face_constructor_z
using Oceananigans.Grids: pop_flat_elements, topology, halo_size, on_architecture
using Oceananigans.Fields
using Oceananigans.Utils: SpecifiedTimes
using Oceananigans.Architectures: arch_array, architecture
using JLD2

import Oceananigans.Fields: set!

abstract type AbstractObservation end

include("normalization.jl")

"""
    SyntheticObservations{F, G, T, P, M} <: AbstractObservation

A time series of synthetic observations generated by Oceananigans.jl's simulations
gridded as Oceananigans.jl fields.
"""
struct SyntheticObservations{F, G, T, P, M, N} <: AbstractObservation
     field_time_serieses :: F
                    grid :: G
                   times :: T
                    path :: P
                metadata :: M
           normalization :: N
end

observation_names(ts::SyntheticObservations) = keys(ts.field_time_serieses)

"""
    observation_names(ts_vector::Vector{<:SyntheticObservations})

Return a Set representing the union of all names in `ts_vector`.
"""
function observation_names(ts_vector::Vector{<:SyntheticObservations})
    names = Set()
    for ts in ts_vector
        push!(names, observation_names(ts)...)
    end

    return names
end

obs_str(ts::SyntheticObservations) = "SyntheticObservations of $(keys(ts.field_time_serieses)) on $(summary(ts.grid))"
obs_str(ts::Vector{<:SyntheticObservations}) = "Vector of SyntheticObservations of $(keys(ts[1].field_time_serieses)) on $(summary(ts[1].grid))"

tupleit(t) = try
    Tuple(t)
catch
    tuple(t)
end

const not_metadata_names = ("serialized", "timeseries")

read_group(group::JLD2.Group) = NamedTuple(Symbol(subgroup) => read_group(group[subgroup]) for subgroup in keys(group))
read_group(group) = group

function with_size(new_size, old_grid)

    topo = topology(old_grid)

    x = cpu_face_constructor_x(old_grid)
    y = cpu_face_constructor_y(old_grid)
    z = cpu_face_constructor_z(old_grid)

    # Remove elements of size and new_halo in Flat directions as expected by grid
    # constructor
    new_size = pop_flat_elements(new_size, topo)
    halo = pop_flat_elements(halo_size(old_grid), topo)

    new_grid = RectilinearGrid(architecture(old_grid), eltype(old_grid);
        size = new_size,
        x = x, y = y, z = z,
        topology = topo,
        halo = halo)

    return new_grid
end

location_guide = Dict(:u => (Face, Center, Center),
                      :v => (Center, Face, Center),
                      :w => (Center, Center, Face))

function infer_location(field_name)
    if field_name in keys(location_guide)
        return location_guide[field_name]
    else
        return (Center, Center, Center)
    end
end

function observation_times(data_path::String)
    file = jldopen(data_path)
    iterations = parse.(Int, keys(file["timeseries/t"]))
    times = [file["timeseries/t/$i"] for i in iterations]
    close(file)
    return times
end

observation_times(observation::SyntheticObservations) = observation.times

function observation_times(obs::Vector)
    @assert all([o.times ≈ obs[1].times for o in obs]) "Observations must have the same times."
    return observation_times(first(obs))
end

function SyntheticObservations(path; field_names,
                               normalize = IdentityNormalization,
                               times = nothing,
                               field_time_serieses = nothing,
                               regrid_size = nothing)

    field_names = tupleit(field_names)

    if isnothing(field_time_serieses)
        field_time_serieses = NamedTuple(name => FieldTimeSeries(path, string(name); times)
                                         for name in field_names)
    end

    grid = first(field_time_serieses).grid
    times = first(field_time_serieses).times
    boundary_conditions = first(field_time_serieses).boundary_conditions

    if !isnothing(regrid_size) # Well, we're gonna regrid stuff

        new_field_time_serieses = Dict()

        # Re-grid the data in `field_time_serieses`
        for (field_name, ts) in zip(keys(field_time_serieses), field_time_serieses)
            #LX, LY, LZ = location(ts[1])
            LX, LY, LZ = infer_location(field_name)
            new_ts = FieldTimeSeries{LX, LY, LZ}(grid, times; boundary_conditions)
        
            # Loop over time steps to re-grid each constituent field in `field_time_series`
            for n = 1:length(times)
                regrid!(new_ts[n], ts[n])
            end
        
            new_field_time_serieses[field_name] = new_ts
        end

        field_time_serieses = NamedTuple(new_field_time_serieses)
    end

    # validate_data(fields, grid, times) # might be a good idea to validate the data...
    file = jldopen(path)
    metadata = NamedTuple(Symbol(group) => read_group(file[group]) for group in filter(n -> n ∉ not_metadata_names, keys(file)))
    close(file)

    normalization = Dict(name => normalize(field_time_serieses[name]) for name in keys(field_time_serieses))

    return SyntheticObservations(field_time_serieses, grid, times, path, metadata, normalization)
end

#####
##### set! for simulation models and observations
#####

function set!(model, ts::SyntheticObservations, index=1)
    # Set initial condition
    for name in keys(fields(model))

        model_field = fields(model)[name]

        if name ∈ keys(ts.field_time_serieses)
            ts_field = ts.field_time_serieses[name][index]
            set!(model_field, ts_field)
        else
            set!(model_field, 0)
        end
    end

    return nothing
end

"""
    column_ensemble_interior(observations::Vector{<:SyntheticObservations}, field_name, time_indices::Vector, N_ens)

Returns an `N_cases × N_ens × Nz` array of the interior of a field `field_name` defined on a 
`OneDimensionalEnsembleGrid` of size `N_cases × N_ens × Nz`, given a list of `SyntheticObservations` objects
containing the `N_cases` single-column fields at time index in `time_index`.
"""
function column_ensemble_interior(observations::Vector{<:SyntheticObservations}, field_name, time_index, ensemble_size)
    zeros_column = zeros(size(observations[1].field_time_serieses[1].grid))
    Nt = length(observation_times(observations))

    batch = []
    for observation in observations
        fts = observation.field_time_serieses
        if field_name in keys(fts) && time_index <= Nt
            push!(batch, interior(fts[field_name][time_index]))
        else
            push!(batch, zeros_column)
        end
    end

    batch = cat(batch..., dims = 2) # (Nbatch, n_z)
    ensemble_interior = cat([batch for i = 1:ensemble_size]..., dims = 1) # (ensemble_size, Nbatch, n_z)

    return ensemble_interior
end

function set!(model, observations::Vector{<:SyntheticObservations}, index = 1)

    for name in keys(fields(model))
    
        model_field = fields(model)[name]
    
        field_ts_data = column_ensemble_interior(observations, name, index, model.grid.Nx)
    
        arch = architecture(model_field)
    
        # Reshape `field_ts_data` to the size of `model_field`'s interior
        reshaped_data = arch_array(arch, reshape(field_ts_data, size(model_field)))
    
        # Sets the interior of field `model_field` to values of `reshaped_data`
        model_field .= reshaped_data
    end

    return nothing
end

struct FieldTimeSeriesCollector{G, D, F, T}
    grid :: G
    field_time_serieses :: D
    collected_fields :: F
    times :: T
end

"""
    FieldTimeSeriesCollector(collected_fields, times; architecture=CPU())

Returns a `FieldTimeSeriesCollector` for `fields` of `simulation`.
`fields` is a `NamedTuple` of `AbstractField`s that are to be collected.
"""
function FieldTimeSeriesCollector(collected_fields, times; architecture=CPU())

    grid = on_architecture(architecture, first(collected_fields).grid)
    field_time_serieses = Dict{Symbol, Any}()

    for name in keys(collected_fields)
        field = collected_fields[name]
        LX, LY, LZ = location(field)
        field_time_series = FieldTimeSeries{LX, LY, LZ}(grid, times)
        field_time_serieses[name] = field_time_series
    end

    # Convert to NamedTuple
    field_time_serieses = NamedTuple(name => field_time_serieses[name] for name in keys(collected_fields))

    return FieldTimeSeriesCollector(grid, field_time_serieses, collected_fields, times)
end

function (collector::FieldTimeSeriesCollector)(simulation)
    for field in collector.collected_fields
        compute!(field)
    end

    current_time = simulation.model.clock.time
    time_index = findfirst(t -> t >= current_time, collector.times)

    for name in keys(collector.collected_fields)
        field_time_series = collector.field_time_serieses[name]
        set!(field_time_series[time_index], collector.collected_fields[name])
    end

    return nothing
end

function initialize_simulation!(simulation, observations, time_series_collector, time_index = 1)
    set!(simulation.model, observations, time_index)

    times = observation_times(observations)

    initial_time = times[time_index]
    simulation.model.clock.time = initial_time
    simulation.model.clock.iteration = 0
    simulation.model.timestepper.previous_Δt = Inf

    # Zero out time series data
    for time_series in time_series_collector.field_time_serieses
        time_series.data .= 0
    end

    simulation.callbacks[:data_collector] = Callback(time_series_collector, SpecifiedTimes(times...))

    simulation.stop_time = times[end]

    return nothing
end

summarize_metadata(::Nothing) = ""
summarize_metadata(metadata) = keys(metadata)

Base.show(io::IO, ts::SyntheticObservations) =
    print(io, "SyntheticObservations with fields $(propertynames(ts.field_time_serieses))", '\n',
              "├── times: $(ts.times)", '\n',
              "├── grid: $(summary(ts.grid))", '\n',
              "├── path: \"$(ts.path)\"", '\n',
              "├── metadata: ", summarize_metadata(ts.metadata), '\n',
              "└── normalization: $(summary(ts.normalization))")

end # module
