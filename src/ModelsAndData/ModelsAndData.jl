module ModelsAndData

using ..OceanTurbulenceParameterEstimation

import Oceananigans.TimeSteppers: time_step!
import Oceananigans.Fields: interpolate
import Oceananigans.Models: AbstractModel
import Oceananigans.Models: HydrostaticFreeSurfaceModel
import Base: length
import StaticArrays: FieldVector

using Oceananigans
using Oceananigans.Grids: Flat, Bounded, Periodic, RegularRectilinearGrid
using Oceananigans: AbstractModel, AbstractEddyViscosityClosure
using Oceananigans.Fields: CenterField, AbstractDataField
using Oceananigans.Grids: Face, Center, AbstractGrid
using Oceananigans.TurbulenceClosures: AbstractTurbulenceClosure

using OrderedCollections, Printf, JLD2
using Base: nothing_sentinel

export
       # file_wrangling
       get_data,
       
       # LESbrary_paths.jl
       LESbrary,
       TwoDaySuite,
       FourDaySuite,
       SixDaySuite,
       GeneralStrat,

       # grids.jl
       ColumnEnsembleGrid,
       XYZGrid,

       # data.jl
       TruthData,
       BatchTruthData,

       # model.jl
       HydrostaticFreeSurfaceModel,
       run_until!,
       initialize_forward_run!,

       # set_fields.jl
       set!,
       column_ensemble_interior,

       # free_parameters.jl
       DefaultFreeParameters,
       get_free_parameters,
       FreeParameters,
       @free_parameters

function initialize_forward_run!(model, data, params, time_index)
    set!(model, params)
    set!(model, data, time_index)
    model.clock.time = data.t[time_index]
    model.clock.iteration = 0
    return nothing
end

function initialize_forward_run!(model, data_batch::BatchTruthData, params, time_indices::Vector)
    set!(model, params)
    set!(model, data_batch, time_indices)
    model.clock.time = 0.0
    model.clock.iteration = 0
    return nothing
end

ensemble_size(model) = model.grid.Nx
batch_size(model) = model.grid.Ny

include("file_wrangling.jl")
include("lesbrary_paths.jl")
include("grids.jl")
include("data.jl")
include("model.jl")
include("set_fields.jl")
include("free_parameters.jl")

end #module