include("EllipsoidFragmentation.jl")
include("ProjectionMapper.jl")
include("SimulationIO.jl")

import .EllipsoidFragmentation
import .ProjectionMapper
import .SimulationIO

import Plots
import DataFrames
import Statistics
using HDF5

function compute_phi2d(config, parents, phi3d, scaling_ratio, overlap)
    println("Computing uncertainty with phi3D = $phi3d, ratio = $scaling_ratio, overlap = $overlap")
    parents_test = deepcopy(parents)
    gen_test = EllipsoidFragmentation.Generator(parents_test, 
                        fill(phi3d, 1),
                        [scaling_ratio],
                        overlap, 
                        config.hard_objects, config.max_placement_attempts)
    
    ells = EllipsoidFragmentation.build_population!(gen_test; verbose=false)
    proj_test = ProjectionMapper.Projector(ells)
    projections_test = ProjectionMapper.project_population(proj_test)

    # Compute error
    counts_2d, children_counts_2d, sigma_counts_2d = ProjectionMapper.count_by_level(proj_test.data, projections_test)

    # println(max_ratio)
    average_n = counts_2d[:xy][0] / children_counts_2d[:xy][0]
    sigma_average = sigma_counts_2d[:xy][0] / sqrt(counts_2d[:xy][0])

    phi_mes2d = - log(average_n) / log(scaling_ratio)
    sigma_phi_mes2d = sigma_average / log(scaling_ratio) / average_n

    println("φ = $phi3d")
    println("φ2D = $phi_mes2d +/- $sigma_phi_mes2d")

    return phi_mes2d, sigma_phi_mes2d
end


config = SimulationIO.load_config("uncertainty_config.toml")
resolution = 10000

phi3D = collect(0.0:0.1:3.0) # from 0 to 3 with 0.1 interval
scaling_ratios = collect(1.2:0.2:3.0)  # from 1.2 to 3 with 0.2 interval
overlap = collect(0.5:0.1:1.0)  # from 0.5 to 1 with 0.1 interval

parents = SimulationIO.create_initial_parents(config; seed=42)

length(ARGS) == 1 || error("Usage : julia compute_tables.jl <klim>")

klim = parse(Int, ARGS[1])

println("klim = ", klim)

h5open("projection_uncertainty.h5", "r+") do f
    for k in eachindex(overlap)
        if k > klim
            for j in eachindex(scaling_ratios)
                for i in eachindex(phi3D)
                    # already computed
                    f["phi2d"][i, j, k] != -1 && continue
    
                    phi2d, sigma_phi2d = compute_phi2d(config, parents, phi3D[i], scaling_ratios[j], overlap[k])
    
                    f["phi2d"][i, j, k] = round.(Int16, phi2d .* resolution) 
                    f["sigma_phi2d"][i, j, k] = round.(UInt16, sigma_phi2d .* resolution)
                end
            end
        end
    end
end