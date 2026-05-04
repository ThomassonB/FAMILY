#!/usr/bin/env julia

"""
Installation script for EllipsoidFragmentation package.
Installs all required Julia packages.
"""

using Pkg

println("="^70)
println("Installing EllipsoidFragmentation dependencies...")
println("="^70)
println()

# List of required packages
packages = [
    "LinearAlgebra",
    "Random", 
    "Statistics",
    "Distributions",
    "Rotations",
    "StaticArrays",
    "ProgressMeter",
    "TOML",
    "CSV",
    "DataFrames",
    "HDF5",
    "JSON",
    "Plots",
    "Trapz",
    "FITSIO"
]

println("Packages to install:")
for pkg in packages
    println("  - $pkg")
end
println()

# Install packages
println("Installing packages...")
println("This may take a few minutes on first installation.")
println()

try
    # Install to default environment (not using Project.toml)
    for pkg in packages
        print("Installing $pkg... ")
        try
            Pkg.add(pkg)
        catch e
            error_msg = string(e)
            if occursin("already installed", error_msg) || occursin("is already installed", error_msg)
                println("(already installed)")
            else
                println("  Error: ", e)
            end
        end
    end
    
    println()
    println("="^70)
    println("Installation completed successfully!")
    println("="^70)
    println()
    println("You can now run simulations with:")
    println("  julia run_simulation.jl config.toml")
    println()
    println("Or go to the example notebook")
    println()
    
catch e
    println()
    println("="^70)
    println("Installation failed!")
    println("="^70)
    println()
    println("Error: ", e)
    println()
    println("Please check your Julia installation and internet connection.")
    println("You can manually install packages with:")
    println("  julia> using Pkg")
    println("  julia> Pkg.add(\"PackageName\")")
end