module ProjectionMapper

import LinearAlgebra
import ProgressMeter
import Trapz

import ..EllipsoidFragmentation: Ellipsoid, collect_ellipsoid_data

export Projector, Mapper
export project_population, build_density_map, project_cube, count_by_level

# ============================================================================
# Projector for 2D projections
# ============================================================================

"""
Projects 3D ellipsoids onto 2D planes (xy, xz, yz).
"""
struct Projector
    data::Dict{String,Vector}
end

function Projector(ellipsoids::Vector{Ellipsoid})
    data = collect_ellipsoid_data(ellipsoids)
    return Projector(data)
end

"""
Project a 3D ellipsoid onto a 2D plane.
Returns (x_center, y_center, semi_major, semi_minor, angle).
"""
function project_ellipse_2d(xo, yo, zo, a, b, c, rotation, axis::Symbol)
    # Define projection matrix based on axis
    if axis == :xy  # Project onto XY plane (view from +Z)
        proj_matrix = [1.0 0.0 0.0; 0.0 1.0 0.0]
        center = [xo, yo]
        abc = [a, b, c]
    elseif axis == :xz  # Project onto XZ plane (view from +Y)
        proj_matrix = [1.0 0.0 0.0; 0.0 0.0 1.0]
        center = [xo, zo]
        abc = [a, c, b]
    elseif axis == :yz  # Project onto YZ plane (view from +X)
        proj_matrix = [0.0 1.0 0.0; 0.0 0.0 1.0]
        center = [yo, zo]
        abc = [b, c, a]
    else
        error("axis must be :xy, :xz, or :yz")
    end
    
    # Project the ellipsoid matrix
    # The ellipsoid in 3D is defined by: (R'(x-xo))' * diag(1/a², 1/b², 1/c²) * (R'(x-xo)) = 1
    # Where R is the rotation matrix
    
    D = LinearAlgebra.Diagonal([1/abc[1]^2, 1/abc[2]^2, 1/abc[3]^2])
    Q = rotation * D * rotation'
    
    # Project Q matrix
    Q_2d = proj_matrix * Q * proj_matrix'
    
    # Compute eigenvalues and eigenvectors
    eigen_result = LinearAlgebra.eigen(Q_2d)
    eigenvalues = eigen_result.values
    eigenvectors = eigen_result.vectors
    
    # Semi-axes are inverse square roots of eigenvalues
    semi_a = 1.0 / sqrt(eigenvalues[1])
    semi_b = 1.0 / sqrt(eigenvalues[2])
    
    # Angle of major axis
    angle = atan(eigenvectors[2, 1], eigenvectors[1, 1])
    
    return (center[1], center[2], semi_a, semi_b, angle)
end

"""
Project all ellipsoids onto three 2D planes.
"""
function project_population(projector::Projector)
    n = length(projector.data["xo"])
    
    projections = Dict(
        :xy => Vector{Tuple{Float64,Float64,Float64,Float64,Float64}}(undef, n),
        :xz => Vector{Tuple{Float64,Float64,Float64,Float64,Float64}}(undef, n),
        :yz => Vector{Tuple{Float64,Float64,Float64,Float64,Float64}}(undef, n)
    )
    
    ProgressMeter.@showprogress "Projecting ellipsoids..." for i in 1:n
        xo = projector.data["xo"][i]
        yo = projector.data["yo"][i]
        zo = projector.data["zo"][i]
        a = projector.data["a"][i]
        b = projector.data["b"][i]
        c = projector.data["c"][i]
        rotation = projector.data["rotation"][i]
        
        for axis in [:xy, :xz, :yz]
            projections[axis][i] = project_ellipse_2d(xo, yo, zo, a, b, c, rotation, axis)
        end
    end
    
    return projections
end

"""
Count ellipsoids by level in 3D and 2D projections.
"""
function count_by_level(data::Dict, projections::Dict)
    levels = unique(data["level"])
    sort!(levels)
    
    counts_3d = Dict{Int,Int}()
    counts_2d = Dict(axis => Dict{Int,Int}() for axis in [:xy, :xz, :yz])
    
    for level in levels
        # 3D count
        counts_3d[level] = count(==(level), data["level"])
        
        # 2D counts (same for all projections since we're just counting objects)
        for axis in [:xy, :xz, :yz]
            idx_level = findall(==(level), data["level"])
            counts_2d[axis][level] = length(idx_level)
        end
    end
    
    return counts_3d, counts_2d
end

# ============================================================================
# Mapper for density maps
# ============================================================================
mutable struct Mapper
    data::Dict{String,Vector}
    grid_resolution::Int
    grid_extent::Float64
    
    # Grid axes
    x_axis::Vector{Float64}
    y_axis::Vector{Float64}
    z_axis::Vector{Float64}
    
    # 3D cube
    total_cube::Union{Array{Float64,3},Nothing}
    
    # Projected cubes
    projected_xy::Union{Matrix{Float64},Nothing}
    projected_xz::Union{Matrix{Float64},Nothing}
    projected_yz::Union{Matrix{Float64},Nothing}
end

function Mapper(ellipsoids::Vector{Ellipsoid}, grid_resolution::Int, grid_extent::Float64)
    data = collect_ellipsoid_data(ellipsoids)
    
    # Create grid axes
    x_axis = range(-grid_extent, grid_extent, length=grid_resolution)
    y_axis = range(-grid_extent, grid_extent, length=grid_resolution)
    z_axis = range(-grid_extent, grid_extent, length=grid_resolution)
    
    return Mapper(data, grid_resolution, grid_extent,
                  collect(x_axis), collect(y_axis), collect(z_axis),
                  nothing, nothing, nothing, nothing)
end

end # module
