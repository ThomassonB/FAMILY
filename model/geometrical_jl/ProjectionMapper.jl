module ProjectionMapper

using Statistics: std
using Statistics: mean
using Graphs

import LinearAlgebra
import ProgressMeter
import Trapz
import LibGEOS
import GeoInterface as GI

import ..EllipsoidFragmentation: Ellipsoid, collect_ellipsoid_data

export Projector, Mapper
export project_population, build_density_map, project_cube, count_by_level

# ============================================================================
# Projector for 2D projections
# ============================================================================
"""
Represents a 2D ellipse with position, size, orientation.
"""
mutable struct Ellipse
    # Position
    xo::Float64
    yo::Float64
    
    # Semi-axes lengths
    a::Float64
    b::Float64
    
    angle::Float64
    
    # Properties
    area::Float64
    level::Int8
    children::Vector{Ellipse}

    # For projections
    polygon::Union{Nothing, LibGEOS.Polygon, LibGEOS.MultiPolygon}
end

"""
Ellipse constructor
"""
function Ellipse(xo, yo, a, b, angle, level, children)
    area = π * a * b
    return Ellipse(xo, yo, a, b, angle, area, level, children, nothing)
end

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
function project_tree(ellipsoid::Ellipsoid, axis::Symbol)
    children = Ellipse[
        project_tree(child, axis) for child in ellipsoid.children
    ]

    return project_ellipse_2d(
        ellipsoid.xo, ellipsoid.yo, ellipsoid.zo,
        ellipsoid.a, ellipsoid.b, ellipsoid.c,
        ellipsoid.level, ellipsoid.rotation,
        children, axis
    )
end

function project_ellipse_2d(xo, yo, zo, a, b, c, l, rotation, children, axis::Symbol)
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
    
    return Ellipse(center[1], center[2], semi_a, semi_b, angle, l, children)
end

"""
Project all ellipsoids onto three 2D planes.
"""
function project_population(projector::Projector)
    n = length(projector.data["xo"])
    
    #projections = Dict(
    #    :xy => Vector{Tuple{Float64,Float64,Float64,Float64,Float64,Int8,Int8}}(undef, n),
    #    :xz => Vector{Tuple{Float64,Float64,Float64,Float64,Float64,Int8,Int8}}(undef, n),
    #    :yz => Vector{Tuple{Float64,Float64,Float64,Float64,Float64,Int8,Int8}}(undef, n)
    #)
    projections = Dict(
        axis => Vector{Ellipse}(undef, n)
        for axis in (:xy, :xz, :yz)
    )
    
    ProgressMeter.@showprogress "Projecting ellipsoids..." for i in 1:n
        xo = projector.data["xo"][i]
        yo = projector.data["yo"][i]
        zo = projector.data["zo"][i]
        a = projector.data["a"][i]
        b = projector.data["b"][i]
        c = projector.data["c"][i]
        l = projector.data["level"][i]
        rotation = projector.data["rotation"][i]
        children = projector.data["children"][i]
        
        #for axis in [:xy, :xz, :yz]
        #    projections[axis][i] = project_ellipse_2d(xo, yo, zo, a, b, c, l, rotation, children, axis)
        #end
        for axis in (:xy, :xz, :yz)
            projected_children = Ellipse[
                project_tree(child, axis)
                for child in children
            ]
        
            projections[axis][i] = project_ellipse_2d(
                xo, yo, zo, a, b, c, l, rotation,
                projected_children, axis
            )
        end
    end
    
    return projections
end

# Convert an ellipse to a polygon ; angle in rad
function ellipse_polygon(e; npoints=128)
    c, s = cos(e.angle), sin(e.angle)

    points = [
        (
            e.xo + e.a * cos(t) * c - e.b * sin(t) * s,
            e.yo + e.a * cos(t) * s + e.b * sin(t) * c
        )
        for t in (2π * j / npoints for j in 0:(npoints - 1))
    ]
    push!(points, first(points))

    coords = join(["$(p[1]) $(p[2])" for p in points], ", ")
    return LibGEOS.readgeom("POLYGON(($coords))")
end

# Extract contours
function contour(geometry)
    return [
        [(Float64(p[1]), Float64(p[2])) for p in ring]
        for ring in GI.coordinates(geometry)
    ]
end

function coordMatrix(polygons)
    coords = Matrix{Float64}(undef, length(polygons), 2)

    for (i, polygon) in enumerate(polygons)
        center = GI.coordinates(LibGEOS.centroid(polygon))
        coords[i, 1] = center[1]
        coords[i, 2] = center[2]
    end

    return coords
end

function sepDistance(coords)
    x = coords[:, 1]
    y = coords[:, 2]

    return hypot.(x .- transpose(x), y .- transpose(y))
end

function merge_overlap(ellipses; npoints=128)
    isempty(ellipses) && return LibGEOS.Polygon[]
    
    polygons = [ellipse_polygon(e; npoints=npoints) for e in ellipses]

    area = mean(e.area for e in ellipses)
    min_separation = sqrt(2 * log(2)) * sqrt(area / π)

    while true
        coords = coordMatrix(polygons)
        distances = sepDistance(coords)
    
        adjacency = distances .< min_separation
        for i in axes(adjacency, 1)
            adjacency[i, i] = false
        end

        !any(adjacency) && return polygons

        graph = SimpleGraph(adjacency)
        cliques = sort(maximal_cliques(graph); by=length, rev=true)

        polygons = [
            reduce(LibGEOS.union, polygons[clique])
            for clique in cliques
        ]
    end
end

function polygon_xy(polygon)
    points = GI.coordinates(polygon)[1]
    return first.(points), getindex.(points, 2)
end

"""
Count ellipsoids by level in 3D and 2D projections.
"""
function append_tree!(population, ellipse::Ellipse)
    push!(population, ellipse)
    for child in ellipse.children
        append_tree!(population, child)
    end
    return population
end

function project_population(roots::AbstractVector{<:Ellipsoid})
    projections = Dict(axis => Ellipse[] for axis in (:xy, :xz, :yz))

    for axis in (:xy, :xz, :yz)
        for root in roots
            append_tree!(projections[axis], project_tree(root, axis))
        end
    end

    return projections
end

function count_by_level(data::Dict, projections::Dict)
    levels = unique(data["level"])
    sort!(levels)
    
    #counts_3d = Dict{Int,Int}()
    #sigma_counts_3d = Dict{Int,Int}()
    
    counts_2d = Dict(axis => Dict{Int,Int}() for axis in [:xy, :xz, :yz])
    sigma_counts_children_2d = Dict(axis => Dict{Int,Float64}() for axis in [:xy, :xz, :yz])
    counts_children_2d = Dict(axis => Dict{Int,Int}() for axis in [:xy, :xz, :yz])
    
    #for level in levels
    #    # 3D count
    #    counts_3d[level] = count(==(level), data["level"])
    #    
    #    # 2D counts (same for all projections since we're just counting objects)
    #    for axis in [:xy, :xz, :yz]
    #        idx_level = findall(==(level), data["level"])
    #        counts_2d[axis][level] = length(idx_level)
    #    end
    #end
    
    #for level in levels
    #    for axis in [:xy, :xz, :yz]
    #        counts_2d[axis][level] = count(e -> e.level == level, projections[axis])
    #        sigma_counts_2d[axis][level] = std([e.children for t in projections[axis] if e.level == level])
    #    end
    #end

    for level in levels
        for axis in (:xy, :xz, :yz)
            parents = filter(e -> e.level == level, projections[axis])
    
            # Number of child after merging, done parents by parents assuming all parents are disconnected
            children_counts = [
                length(merge_overlap(parent.children))
                for parent in parents
            ]
    
            counts_2d[axis][level] = length(parents)
            #sigma_counts_2d[axis][level] = std([e.children for t in projections[axis] if e.level == level]) length(children_counts) >= 2 ? std(children_counts) : NaN
            
            counts_children_2d[axis][level] = length(children_counts) >= 2 ? sum(children_counts) : NaN 
            sigma_counts_children_2d[axis][level]    = length(children_counts) >= 2 ? std(children_counts) : NaN
        end
    end
    
    return counts_2d, counts_children_2d, sigma_counts_children_2d
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
