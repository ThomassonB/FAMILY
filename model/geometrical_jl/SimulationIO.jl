module SimulationIO

import TOML
import CSV
import DataFrames
import HDF5
import JSON
# import Dates

import Random
import Distributions
import StaticArrays

import ..EllipsoidFragmentation: Ellipsoid, Generator, collect_ellipsoid_data

export Config, load_config, save_results, load_results
export create_initial_parents, create_generator

# ============================================================================
# Configuration
# ============================================================================
struct Config
    # Simulation parameters
    n_initial_parents::Int
    initial_volume_range::Float64
    n_vertices::Int
    overlap_threshold::Float64
    hard_objects::Float64
    max_placement_attempts::Int
    
    # Scale parameters
    scale_method::String
    n_levels::Int
    scaling_ratio::Float64
    scale_list::Union{Vector{Float64},Nothing}
    
    # Fragmentation parameters
    fragmentation_model::String
    fragmentation_rate::Float64
    fragmentation_function::Union{String,Nothing}
    
end

function load_config(config_file::String)
    if !isfile(config_file)
        error("Configuration file not found: $config_file")
    end
    
    toml = TOML.parsefile(config_file)
    
    # Extract parameters with defaults
    sim = get(toml, "simulation", Dict())
    scales = get(toml, "scales", Dict())
    frag = get(toml, "fragmentation", Dict())
    proj = get(toml, "projection", Dict())
    out = get(toml, "output", Dict())
    sweep = get(toml, "parameter_sweep", Dict())
    
    return Config(
        # Simulation
        get(sim, "n_initial_parents", 10),
        get(sim, "initial_volume_range", 10.0),
        get(sim, "n_vertices", 32),
        get(sim, "overlap_threshold", 0.9),
        get(sim, "hard_objects", 1e-3),
        get(sim, "max_placement_attempts", 10000),
        
        # Scales
        get(scales, "method", "ratio"),
        get(scales, "n_levels", 2),
        get(scales, "scaling_ratio", 2.0),
        get(scales, "scale_list", nothing),
        
        # Fragmentation
        get(frag, "model", "constant"),
        get(frag, "fragmentation_rate", 1.1),
        get(frag, "fragmentation_function", nothing),
    )
end

# ============================================================================
# Initial Population Creation
# ============================================================================
function create_initial_parents(config::Config; seed=nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    
    n = config.n_initial_parents
    range = config.initial_volume_range

    # Generate random ellipsoid parameters
    a = Random.rand(Distributions.Normal(1.0, 0.1), n)
    a = max.(a, 0.1)  # Ensure positive
    
    b = Random.rand(Distributions.Uniform(0.7, 1.0), n) .* a
    c = Random.rand(Distributions.Uniform(1.0, 1.3), n) .* a
    
    # Random positions
    xo = Random.rand(Distributions.Uniform(-range, range), n)
    yo = Random.rand(Distributions.Uniform(-range, range), n)
    zo = Random.rand(Distributions.Uniform(-range, range), n)
    
    # Random orientations
    prec = Random.rand(Distributions.Uniform(0, 90), n)
    nut = Random.rand(Distributions.Uniform(0, 90), n)
    gir = Random.rand(Distributions.Uniform(0, 90), n)
    
    # Create ellipsoids
    parents = [Ellipsoid(xo[i], yo[i], zo[i], a[i], b[i], c[i],
                         prec[i], nut[i], gir[i]; level=0, parent_idx=i)
               for i in 1:n]
    
    return parents
end

function set_scaling_ratios(config::Config)
    if config.scale_method == "ratio"
        scaling_ratios = fill(config.scaling_ratio, config.n_levels)
    elseif config.scale_method == "list" && !isnothing(config.scale_list)
        # Convert scale list to ratios
        scaling_ratios = [config.scale_list[i] / config.scale_list[i+1] 
                         for i in 1:length(config.scale_list)-1]
    else
        error("Invalid scale configuration")
    end
    return scaling_ratios
end

function create_generator(config::Config, parents::Vector{Ellipsoid})
    # Determine scaling ratios
    scaling_ratios = set_scaling_ratios(config)
    
    # Determine fragmentation rates
    if config.fragmentation_model == "constant"
        fragmentation_rates = fill(config.fragmentation_rate, length(scaling_ratios))
    elseif config.fragmentation_model == "function" && !isnothing(config.fragmentation_function)
        # Evaluate function for each level
        func = eval(Meta.parse("level -> " * config.fragmentation_function))
        fragmentation_rates = [func(level) for level in 1:length(scaling_ratios)]
    else
        error("Invalid fragmentation configuration")
    end
    
    return Generator(parents, fragmentation_rates, scaling_ratios,
                    config.overlap_threshold, config.hard_objects,
                    config.max_placement_attempts)
end

end # module
