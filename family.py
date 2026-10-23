import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


reader = lambda x: pd.read_csv(x, delimiter=",", comment="#")

from analysis import analyse
from analysis import network_utility as utility
from analysis import fragmeasurement as fm
from analysis import plotter

def main():
    parser = argparse.ArgumentParser(description="FAMILY script to measure fragmentation spatial rate")
    parser.add_argument("dir", help="Directory where you have your configuration files for your catalogs config.toml")
    parser.add_argument("--overlap", type=float, default=0.75,
                        help="Minimum overlap area before two objects are considered disjointed.")

    args = parser.parse_args()
    
    path = Path(args.dir).resolve()
    print(os.getcwd())
    print(args.dir)
    print(path)

    # Read the data
    dataset = analyse.DataSet(
        analyse.Data(config, reader_to_df=reader)
        for config in path.glob("*.toml")
    )

    for name, data in dataset.items():
        print(f"sample {name} has {len(data.df)} elements")
        
    print(f"total #elements = {sum(data.df.shape[0] for data in dataset)}")

    # Reconstruct parentality
    network = analyse.Network(dataset, min_overlap=args.overlap)
    network.extractStructures()
    structure_table = network.getStructuresTable()

    # Phi measurement
    fractalities = [struct.fractality for struct in network.structures]
    print(f"Mean : {np.mean(fractalities)}\nMedian : {np.median(fractalities)}")
    
    fractalities = [struct.fractality for struct in network.structures if struct.mode.name == "HIERARCHICAL"]
    print(f"Mean (Hierarchical only) : {np.mean(fractalities)}\nMedian (Hierarchical only) : {np.median(fractalities)}")

    # uncertainties
    # missing nodes
    virtual = utility.virtualNodes(network.network, network.levels)
    network_filled = analyse.Network(network.data, min_overlap=network.min_overlap, graph=virtual)
    utility.fractality(network_filled.network)
    
    network_filled.extractStructures()
    
    fractalities = [struct.fractality for struct in network.structures if struct.mode.name == "HIERARCHICAL"]
    fractalities_filled = [struct.fractality for struct in network_filled.structures if struct.mode.name == "HIERARCHICAL"]
    
    print("median with virtual nodes", np.median(fractalities))
    print("median without virtual nodes", np.median(fractalities_filled))

    # phi 2D
    f_up = np.median(fractalities_filled)
    f_low = np.median(fractalities)
    
    fractality, delta_fractality = fm.estimated_f(f_up, f_low)
    phi2D = fm.phi2D(fractality)
    delta_phi2D = fm.delta_phi2D(fractality, delta_fractality)
    
    print(f"Fragmentation rate in 2D : {phi2D}+/-{delta_phi2D}")

    # phi 3D
    relative_error = 1
    delta_relative_error = 0.1
    
    phi3D = fm.phi3D(phi2d, relative_error=relative_error)
    delta_phi3D = fm.delta_phi3D(phi2d, delta_phi2d, relative_error, delta_relative_error)
    
    print(f"Fragmentation rate in 3D : {phi3D}+/-{delta_phi3D}")
    
    # plots
    distance = network.data["L01"].distance
    x, meany, stdy, y = plotter.getFragCurve(network, distance, mode = "HIERARCHICAL")

    plotter.InspectNetwork(network)


if __name__ == "__main__":
    main()