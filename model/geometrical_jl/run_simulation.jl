#!/usr/bin/env julia

"""
Main script for running 3D ellipsoid fragmentation simulations.

Usage:
    julia run_simulation.jl [config_file]
    julia run_simulation.jl --sweep [config_file]
    julia run_simulation.jl --help
"""

# Load all modules
include("EllipsoidFragmentation.jl")
include("ProjectionMapper.jl")
include("SimulationIO.jl")

import .EllipsoidFragmentation
import .ProjectionMapper
import .SimulationIO

# import Printf
import Dates

function print_banner()
    println("="^70)
    println(" 3D Ellipsoid Fragmentation Monte-Carlo")
    println("="^70)
    println()
end

function run_simulation(config_file::String; verbose=true, seed=nothing)
    # Load configuration
    if verbose
        println("Loading configuration from: $config_file")
    end
    config = SimulationIO.load_config(config_file)
    
    # Create output directory
    mkpath(config.output_dir)
    
    # Create initial parents
    if verbose
        println("\nCreating $(config.n_initial_parents) initial parent ellipsoids...")
    end
    parents = SimulationIO.create_initial_parents(config; seed=seed)
    
    # Create generator
    gen = SimulationIO.create_generator(config, parents)
    
    if verbose
        println("\nSimulation parameters:")
        println("  Fragmentation rates: ", gen.fragmentation_rates)
        println("  Scaling ratios: ", gen.scaling_ratios)
        println("  Overlap threshold: ", config.overlap_threshold)
        println()
    end
    
    # Build population
    start_time = time()
    if verbose
        println("Building ellipsoid population...")
    end
    ellipsoids = EllipsoidFragmentation.build_population!(gen; verbose=verbose)
    build_time = time() - start_time
    
    if verbose
        println("\nPopulation built successfully!")
        println("  Total ellipsoids: ", length(ellipsoids))
        println("  Build time: $(round(build_time, digits=2)) seconds")
        
        # Count by level
        levels = [e.level for e in ellipsoids]
        for level in sort(unique(levels))
            count = sum(levels .== level)
            println("    Level $level: $count ellipsoids")
        end
    end
    
    # Project population
    if verbose
        println("\nProjecting population onto 2D planes...")
    end
    projector = ProjectionMapper.Projector(ellipsoids)
    projections = ProjectionMapper.project_population(projector)
    
    # Count objects by level
    counts_3d, counts_2d = ProjectionMapper.count_by_level(projector.data, projections)
    
    if verbose
        println("\n2D Projection counts:")
        println("  XY plane: ", length(projections[:xy]))
        println("  XZ plane: ", length(projections[:xz]))
        println("  YZ plane: ", length(projections[:yz]))
    end

    if verbose
        println("\n" * "="^70)
        println("Simulation completed successfully!")
        println("="^70)
    end
    
    return ellipsoids, projections, metadata
end

"""
Print usage information.
"""
function print_help()
    println("""
    3D Ellipsoid Fragmentation Simulation
    
    Usage:
        julia run_simulation.jl [options] [config_file]
    
    Options:
        --seed N        Set random seed for reproducibility
        --quiet         Suppress verbose output
        --help          Show this help message
    
    Arguments:
        config_file     Path to TOML configuration file
                       (default: config.toml)
    
    Examples:
        # Run single simulation
        julia run_simulation.jl config.toml

        # Run with specific seed
        julia run_simulation.jl --seed 42 config.toml
    
    For more information, see the README.md file.
    """)
end

function main()
    # Parse command-line arguments
    args = ARGS
    
    sweep_mode = false
    verbose = true
    seed = nothing
    config_file = "config.toml"
    
    i = 1
    while i <= length(args)
        if args[i] == "--quiet"
            verbose = false
        elseif args[i] == "--seed"
            i += 1
            seed = parse(Int, args[i])
        elseif args[i] == "--help" || args[i] == "-h"
            print_help()
            return
        else
            config_file = args[i]
        end
        i += 1
    end
    
    # Print banner
    if verbose
        print_banner()
    end
    
    try
        run_simulation(config_file; verbose=verbose, seed=seed)
    catch e
        if isa(e, InterruptException)
            println("\n\nSimulation interrupted by user")
        else
            println("\nError occurred during simulation:")
            println(e)
            if verbose
                rethrow(e)
            end
        end
    end
end

# Run main if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
