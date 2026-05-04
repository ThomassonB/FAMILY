"""
    EllipsoidFragmentation

Julia module for simulating 3D ellipsoid fragmentation with hierarchical nesting
and quantifying projection errors.
"""
module EllipsoidFragmentation

# import LinearAlgebra
# import Statistics
import Random
import Distributions
import Rotations
import StaticArrays
import ProgressMeter

export Ellipsoid, Generator
export build_population!, collect_ellipsoid_data

# ============================================================================
# Data Structures
# ============================================================================

"""
Represents a 3D ellipsoid with position, size, orientation, and hierarchical structure.
"""
mutable struct Ellipsoid
    # Position
    xo::Float64
    yo::Float64
    zo::Float64
    
    # Semi-axes lengths
    a::Float64
    b::Float64
    c::Float64
    
    # Rotation matrix (3x3)
    rotation::StaticArrays.SMatrix{3,3,Float64,9}
    
    # Hierarchical information
    level::Int
    parent_idx::Union{Int,Nothing}
    
    # Children ellipsoids
    children::Vector{Ellipsoid}
    
    # Properties
    volume::Float64
end

"""
Ellipsoid constructor with Euler angles
"""
function Ellipsoid(xo, yo, zo, a, b, c, prec, nut, gir; level=0, parent_idx=nothing)
    # Convert Euler angles (ZXZ convention) to rotation matrix
    rotation = Rotations.RotZXZ(deg2rad(prec), deg2rad(nut), deg2rad(gir))
    rot_matrix = StaticArrays.SMatrix{3,3}(rotation)
    
    volume = (4/3) * π * a * b * c
    
    return Ellipsoid(xo, yo, zo, a, b, c, rot_matrix, level, parent_idx, 
                     Ellipsoid[], volume)
end

"""
Check if a point is inside the ellipsoid
"""
function is_inside(ell::Ellipsoid, point::AbstractVector)
    # Transform to ellipsoid local coordinates
    p_local = ell.rotation' * (point - [ell.xo, ell.yo, ell.zo])
    
    # Ellipsoid equation
    term = (p_local[1] / ell.a)^2 + (p_local[2] / ell.b)^2 + (p_local[3] / ell.c)^2
    
    return term <= 1.0
end

"""
Test if child ellipsoid is within parent ellipsoid
"""
function sample_points_child(child::Ellipsoid; sampling=1000)
    x = (2Random.rand(sampling) .- 1) .* child.a
    y = (2Random.rand(sampling) .- 1) .* child.b
    z = (2Random.rand(sampling) .- 1) .* child.c

    inside_child = (x ./ child.a).^2 .+
                   (y ./ child.b).^2 .+
                   (z ./ child.c).^2 .<= 1.0

    coords_local_child = vcat(x', y', z')  # 3×N
    coords_local_child = coords_local_child[:, inside_child]

    # Transform to absolute coordinates
    coords_abs = child.rotation * coords_local_child
    coords_abs[1, :] .+= child.xo
    coords_abs[2, :] .+= child.yo
    coords_abs[3, :] .+= child.zo

    return coords_abs
end

function within(child::Ellipsoid, parent::Ellipsoid; minimum=0.9, sampling=1000)
    coords_abs = sample_points_child(child; sampling=sampling)

    # Transform to parent coordinates
    coords_local_parent = parent.rotation' * (
        coords_abs .- [parent.xo, parent.yo, parent.zo]
    )

    inside_parent = (coords_local_parent[1, :] ./ parent.a).^2 .+
                    (coords_local_parent[2, :] ./ parent.b).^2 .+
                    (coords_local_parent[3, :] ./ parent.c).^2 .<= 1.0

    return count(inside_parent) / size(coords_abs, 2) >= minimum
end

"""
Check if two ellipsoids intersect
"""
function intersects(ell1::Ellipsoid, ell2::Ellipsoid; hard_threshold=1e-3)
    # Quick distance check
    dist = sqrt((ell1.xo - ell2.xo)^2 + (ell1.yo - ell2.yo)^2 + (ell1.zo - ell2.zo)^2)
    max_extent = max(ell1.a, ell1.b, ell1.c) + max(ell2.a, ell2.b, ell2.c)
    
    if dist > max_extent
        return false
    end
    
    # Sample-based intersection test
    return within(ell1, ell2; minimum=hard_threshold, sampling=1000)
end

# ============================================================================
# Generator Class
# ============================================================================

"""
Hhierarchical generation of ellipsoid populations with fragmentation
"""
struct Generator
    initial_parents::Vector{Ellipsoid}
    fragmentation_rates::Vector{Float64}
    scaling_ratios::Vector{Float64}
    overlap_threshold::Float64
    hard_objects::Float64
    max_attempts::Int
end

"""
Create a pool of candidate child ellipsoids.
"""
function create_child_candidates(parent::Ellipsoid, size::Float64, limit::Int)
    # Generate random sizes
    a = Random.rand(Distributions.Normal(size, 0.1 * size), limit)
    b = Random.rand(Distributions.Uniform(0.7, 1.0), limit) .* a
    c = Random.rand(Distributions.Uniform(1.0, 1.3), limit) .* a
    
    # Generate random orientations (Euler angles)
    prec = Random.rand(Distributions.Uniform(0, 90), limit)
    nut = Random.rand(Distributions.Uniform(0, 90), limit)
    gir = Random.rand(Distributions.Uniform(0, 90), limit)
    
    # Generate positions in parent's coordinate system
    xo = (2Random.rand(limit) .- 1) .* a
    yo = (2Random.rand(limit) .- 1) .* b
    zo = (2Random.rand(limit) .- 1) .* c
    
    # Transform to absolute coordinates
    positions = parent.rotation * vcat(xo', yo', zo')
    xo = positions[1, :] .+ parent.xo
    yo = positions[2, :] .+ parent.yo
    zo = positions[3, :] .+ parent.zo
    
    return (xo, yo, zo, a, b, c, prec, nut, gir)
end

"""
Place children ellipsoids inside parent
"""
function place_children!(parent::Ellipsoid, n_children::Int, gen::Generator, level::Int)
    if n_children == 0
        return
    end
    
    size = 1.0 / gen.scaling_ratios[level]^level
    sep = size^2
    
    # Minimum radial distance for packing
    rmin = (n_children != 1) ? sep / 2 : 0.0
    
    count = 0
    max_iterations = gen.max_attempts
    candidate_idx = gen.max_attempts
    xo, yo, zo, a, b, c, prec, nut, gir = create_child_candidates(parent, size, gen.max_attempts)
    
    while length(parent.children) < n_children && count < max_iterations
        # Generate new candidates if needed
        if candidate_idx >= gen.max_attempts
            xo, yo, zo, a, b, c, prec, nut, gir = create_child_candidates(parent, size, gen.max_attempts)
            candidate_idx = 0
        end

        candidate_idx += 1
        if candidate_idx > gen.max_attempts
            count += 1
            continue
        end
        
        # Check radial exclusion
        ro_sq = (xo[candidate_idx] - parent.xo)^2 + 
                (yo[candidate_idx] - parent.yo)^2 + 
                (zo[candidate_idx] - parent.zo)^2
        
        if ro_sq < rmin^2
            count += 1
            continue
        end
        
        # Create candidate child
        child = Ellipsoid(
            xo[candidate_idx], yo[candidate_idx], zo[candidate_idx],
            a[candidate_idx], b[candidate_idx], c[candidate_idx],
            prec[candidate_idx], nut[candidate_idx], gir[candidate_idx];
            level=parent.level + 1, parent_idx=parent.parent_idx
        )

        # Check if child is within parent
        if !within(child, parent; minimum=gen.overlap_threshold)
            count += 1
            continue
        end
        
        # Check collision with siblings
        collides = false
        for sibling in parent.children
            if intersects(child, sibling; hard_threshold=gen.hard_objects)
                collides = true
                break
            end
        end
        
        if !collides
            push!(parent.children, child)
        end
        
        count += 1
    end
    
    #if length(parent.children) < n_children
    #    @warn "Could only place $(length(parent.children))/$n_children children after $max_iterations attempts"
    #end
end

"""
Build the complete hierarchical population of ellipsoids
"""
function build_population!(gen::Generator; verbose=true)
    n_levels = length(gen.fragmentation_rates)
    
    # Start with initial parents
    current_level = gen.initial_parents
    all_ellipsoids = copy(gen.initial_parents)
    
    for level in 1:n_levels
        if verbose
            println("Building level $level with $(length(current_level)) parents...")
        end
        
        next_level = Ellipsoid[]
        phi = gen.fragmentation_rates[level]
        ratio = gen.scaling_ratios[level]
        
        avg_children = ratio^phi
        
        ProgressMeter.@showprogress for parent in current_level
            # Determine number of children (binary distribution around mean)
            n_floor = floor(Int, avg_children)
            n_ceil = ceil(Int, avg_children)
            p_ceil = avg_children - n_floor
            
            n_children = Random.rand() < p_ceil ? n_ceil : n_floor
            
            # Place children
            place_children!(parent, n_children, gen, level)
            
            # Add to next level
            append!(next_level, parent.children)
        end
        
        if verbose
            println("  Created $(length(next_level)) children")
        end
        
        append!(all_ellipsoids, next_level)
        current_level = next_level
    end
    
    return all_ellipsoids
end

"""
Collect all ellipsoid parameters into arrays for export.
"""
function collect_ellipsoid_data(ellipsoids::Vector{Ellipsoid})
    n = length(ellipsoids)
    
    data = Dict(
        "xo" => [e.xo for e in ellipsoids],
        "yo" => [e.yo for e in ellipsoids],
        "zo" => [e.zo for e in ellipsoids],
        "a" => [e.a for e in ellipsoids],
        "b" => [e.b for e in ellipsoids],
        "c" => [e.c for e in ellipsoids],
        "level" => [e.level for e in ellipsoids],
        "volume" => [e.volume for e in ellipsoids],
        "rotation" => [e.rotation for e in ellipsoids]
    )
    
    return data
end

end # module
