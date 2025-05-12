from copy import deepcopy
from sys import getsizeof
import numpy as np

import shapely.geometry as shp
from scipy.spatial import ConvexHull

import polygons_utility as pu
import statfrag as statfrag

import networkx as nx
from shapely.ops import unary_union
import pandas as pd

from tqdm import tqdm

def rotateEllipsoid(coords, rotation):
    arr = np.ravel(coords)
    arr = np.reshape(arr, (3, len(arr)//3))
    return np.matmul(rotation, arr)

class Centroid:
    def __init__(self, xo, yo, zo):
        self.xo = xo
        self.yo = yo
        self.zo = zo

class Ellipsoid:
    def __init__(self, xo, yo, zo, a, b, c, rotation=(0, 0, 0), level=0, sidx=None, **kwargs):
        from scipy.spatial.transform import Rotation

        if isinstance(rotation, tuple):
            self.rotation = Rotation.from_euler('ZXZ', rotation, degrees=True).as_matrix()
        else:
            self.rotation = rotation
            
        self.centroid = Centroid(xo, yo, zo)
        self.a = a
        self.b = b
        self.c = c
        
        self.level = level
        self.volume = 4/3*np.pi*self.a*self.b*self.c

        self.sidx = sidx
        self.ellchild = []

    def __iter__(self):
        yield self
        for ellipse in self.ellchild:
            yield from ellipse.__iter__()

    def _set_new_child(self, CoordModel, size, limit):
        # create a pool of coordinates
        a = np.random.normal(1 * size, 0.1 * size, size=limit)
        b = np.random.uniform(low=0.7, high=1, size=limit) * a
        c = np.random.uniform(low=1, high=1.3, size=limit) * a

        prec, nut, gir = np.random.uniform(0, 90, size=(3, limit))

        xo, yo, zo = CoordModel.get_xyz(a, b, c, size=limit)

        xyz = rotateEllipsoid([xo, yo, zo], self.rotation)
        xo = xyz[0] + self.centroid.xo
        yo = xyz[1] + self.centroid.yo
        zo = xyz[2] + self.centroid.zo
        return xo, yo, zo, a, b, c, prec, nut, gir

    def getContour(self, N = 32):
        u = np.linspace(0, 2 * np.pi, N)
        v = np.linspace(0, np.pi, N)
        x = self.a * np.outer(np.cos(u), np.sin(v))
        y = self.b * np.outer(np.sin(u), np.sin(v))
        z = self.c * np.outer(np.ones(np.size(u)), np.cos(v))
        
        xyz = rotateEllipsoid([x,y,z], self.rotation)
        xyz[0] += self.centroid.xo
        xyz[1] += self.centroid.yo
        xyz[2] += self.centroid.zo
        return xyz
    
    def isInside(self, point, relative=False):
        if relative:
            A = ( (point[0] - self.centroid.xo) / self.a )**2
            B = ( (point[1] - self.centroid.yo) / self.b )**2
            C = ( (point[2] - self.centroid.zo) / self.c )**2
        else:
            A = ( point[0] / self.a )**2
            B = ( point[1] / self.b )**2
            C = ( point[2] / self.c )**2
        return A + B + C <= 1

    
    def within(self, other, minimum, sampling = 1000, verbose=False):
        #test if centroid separation > 2x max abc to proceed, else : return False
        #test if x,y,z,x',y',z' exist such that equation ell1 == 1 == equation ell2
        
        x = np.random.uniform(low=-other.a, high=other.a, size=sampling)
        y = np.random.uniform(low=-other.b, high=other.b, size=sampling)
        z = np.random.uniform(low=-other.c, high=other.c, size=sampling)
        
        idx = other.isInside([x, y, z])
        arr = np.array([x[idx], y[idx], z[idx]])

        # base 0
        point = np.matmul(other.rotation, arr)
        #print("shape", point.shape, "=", other.rotation.shape, arr.shape)
        point[0] += other.centroid.xo
        point[1] += other.centroid.yo
        point[2] += other.centroid.zo
        
        #in self's pov
        point[0] += - self.centroid.xo
        point[1] += - self.centroid.yo
        point[2] += - self.centroid.zo
        point = np.matmul(np.linalg.inv(self.rotation), point)

        inside = self.isInside(point)
        #idx = self.isInside(point, relative=True)
        
        if verbose:
            # in abs referential
            point = np.matmul(other.rotation, arr)
            point[0] += other.centroid.xo
            point[1] += other.centroid.yo
            point[2] += other.centroid.zo
            print("Proportion inside:", sum(inside)/len(inside))
            print("Inside = ", sum(inside)/len(inside) >= minimum)
            
            in_point = np.array([point[0][inside], point[1][inside], point[2][inside]])
            out_point = np.array([point[0][~inside], point[1][~inside], point[2][~inside]])
             
            return in_point, out_point, inside
        
        return sum(inside)/len(inside) >= minimum

    def children(self, number_to_place, size, sep, CoordModel, limit = 10_000, hard_objects=1e-3, **kwargs):
        xo, yo, zo, a, b, c, prec, nut, gir = self._set_new_child(CoordModel, size, limit)
        ro = xo ** 2 + yo ** 2 + zo ** 2

        if number_to_place != 1:
            rmin = sep / 4
        else:
            rmin = 0

        r_exclusion = ro >= rmin
        count = 0
        while (len(self.ellchild) < number_to_place and count < limit) or len(self.ellchild)==0:

            # resample test children if too many tries
            if count >= limit:
                count = 0
                xo, yo, zo, a, b, c, prec, nut, gir = self._set_new_child(CoordModel, size, limit)

                ro = xo ** 2 + yo ** 2 + zo ** 2
                rmin = 0
                r_exclusion = ro >= rmin

            # coordinates check 1) inside parent 2) hard sphere 3) exclusion area rmin (help packing)
            if r_exclusion[count]:
                ellchild = Ellipsoid(xo[count], yo[count], zo[count], 
                                     a[count], b[count], c[count],
                                     rotation=(prec[count], nut[count], gir[count]),
                                     level = self.level + 1, sidx = self.sidx)

                # check if children intersect
                hard = [not ellchild.within(other, minimum = hard_objects) for other in self.ellchild]
                if self.within(ellchild, **kwargs) and np.all(hard):
                    self.ellchild.append(ellchild)
                    
            count += 1

class Generator:
    def __init__(self, initial_parents, fragmentation_rates, scaling_ratios, overlap, DiscreteModel, CoordModel):
        self.fragmentation_rates = fragmentation_rates
        self.scaling_ratios = scaling_ratios
        if len(self.fragmentation_rates) != len(self.scaling_ratios):
            raise ("Fragmentation rates and scaling ratios size don't match:\n"
                   f"\tFragmentation rates is size {len(self.fragmentation_rates)}\n"
                   f"\tScaling ratios is size {len(self.scaling_ratios)}")

        self.overlap = overlap

        self.initial_parents = initial_parents

        self.DiscreteModel = DiscreteModel
        self.CoordModel = CoordModel

    """
    def mapPopulation_3D(self, initial_parents, fragmentation_rates, scaling_ratios, overlaps, Npoly = 64):
        PHIS, R, OVER = np.meshgrid(fragmentation_rates, scaling_ratios, overlaps)
        
        for phi3D, ratio, over in zip(PHIS.flatten(), R.flatten(), OVER.flatten()):
            N = ratio ** phi3D
            ellc = copy.deepcopy(initial_parents)
            size = 1/ratio
            sep = size**2
            
            #children = []
            for ell in tqdm(ellc):
                
                n = self.selectFragmentNumber(N)
                ell.children(number_to_place = n, size = size, sep = sep, minimum = over)
                #children += ell.ellchild
                
            #total = ellc + children
            self.buildPopulation_3D(parent_list, fragmentation_rates, scaling_ratios, overlap, level=0, Npoly=64)

            #self.save_population(total, phi3D, ratio, over)
    """

    def buildPopulation_3D(self, parent_list=None, level = 0, Npoly = 64):
        if parent_list is None:
            parent_list = self.initial_parents

        if level < len(self.scaling_ratios):
            phi = self.fragmentation_rates[level]
            ratio = self.scaling_ratios[level]

            average_number = ratio ** phi
            size = 1/ratio
            sep = size**2

            children = []
            print(f"\n\t Building generation {level + 1}/{len(self.scaling_ratios)} \n\t \\phi = {phi} and ratio = {ratio}")
            print(f"\t Average of {np.round(average_number, decimals=2)} fragments per parent")
            print(f"\t {len(parent_list)} parents to handle")

            for parent in tqdm(parent_list):
                #n = self.selectFragmentNumber(number)
                self.DiscreteModel.set_mean(average_number)
                n = self.DiscreteModel.get_number()
                parent.children(number_to_place = n, size = size, sep = sep, CoordModel=self.CoordModel, minimum = self.overlap)

                children += parent.ellchild

            self.buildPopulation_3D(parent_list=children, level = level + 1, Npoly=Npoly)

    def save_population(self, file_name, format="csv", save_path='./', comments=""):
        if format == "csv":
            self._save_population_csv(file_name, save_path=save_path, comments=comments)
        elif format == "hdf5":
            pass
        else:
            raise ("format arg not recognised. Please enter one of the following format as a string:\n"
                   "\t csv\n"
                   "\t hdf5")

    def _save_population_csv(self, file_name, save_path='./', comments=""):

        data = {
            "xo": [],
            "yo": [],
            "zo": [],
            "a": [],
            "b": [],
            "c": [],
            "rot": [],
            "level": [],
            "structure": [],
        }

        for root in self.initial_parents:
            for ellipse in root:
                data["xo"].append(ellipse.centroid.xo)
                data["yo"].append(ellipse.centroid.yo)
                data["zo"].append(ellipse.centroid.zo)
                data["a"].append(ellipse.a)
                data["b"].append(ellipse.b)
                data["c"].append(ellipse.c)
                data["rot"].append(ellipse.rotation)
                data["level"].append(ellipse.level)
                data["structure"].append(ellipse.sidx)

        pd_data = pd.DataFrame(data)

        meta_data = {"Comments":comments,
                     "Fragmentation rates":self.fragmentation_rates,
                     "Scaling ratios":self.scaling_ratios,
                     "Overlap":self.overlap}

        #phi3D = np.round(self.fragmentation_rates, decimals=2)
        #ratio = np.round(ratio, decimals=2)
        #overlap = np.round(overlap, decimals=2)
        #name = "phi3D_" + str(phi3D).replace('.', 'p') + "r_" + str(ratio).replace('.', 'p') + "over_" + str(
        #    overlap).replace('.', 'p')

        with open(save_path + file_name + '.txt', 'w') as file:
            for key, value in meta_data.items():
                file.write(key + ' >>> '+ str(value) + '\n\n')

        pd_data.to_csv(save_path + file_name + ".csv", index=False)

    def plot(self, colors={0:"blue", 1:"red", 2:"green"}, Nvertices = 32):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_box_aspect((20, 20, 20))

        ax.set_xlabel('$X$', rotation=150)
        ax.set_ylabel('$Y$')
        ax.set_zlabel('Z', rotation=60)

        # Hide grid lines
        ax.grid(False)

        # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        for parents in self.initial_parents:
            for ellipses in parents:
                xyz = ellipses.getContour(N=Nvertices)
                x = np.reshape(xyz[0], (Nvertices, Nvertices))
                y = np.reshape(xyz[1], (Nvertices, Nvertices))
                z = np.reshape(xyz[2], (Nvertices, Nvertices))
                ax.plot_surface(x, y, z, color=colors[ellipses.level], alpha=0.5)

        plt.show()

class Projector:
    def __init__(self, ellipses_data, metadata=None):
        self.data = ellipses_data
        self.metadata = metadata

        self.levels = self.data["level"].unique()

        self.polygons_x = {level:[] for level in self.levels}
        self.polygons_y = {level: [] for level in self.levels}
        self.polygons_z = {level: [] for level in self.levels}

        self.merged_polygons_x = {level:[] for level in self.levels}
        self.merged_polygons_y = {level: [] for level in self.levels}
        self.merged_polygons_z = {level: [] for level in self.levels}

    def project_population(self):
        for level in self.levels:
            sub_df = self.data['level'] == level
            
            ellist = [Ellipsoid(self.data["xo"][i],
                                self.data["yo"][i],
                                self.data["zo"][i],
                                self.data["a"][i],
                                self.data["b"][i],
                                self.data["c"][i],
                                rotation = self.data["rot"][i],
                                level=self.data["level"][i])
                    for i in self.data[sub_df].index]
            
            self.polygons_x[level] = self.extract_polygons(ellist, Npoly = 64, axis="xy")
            self.merged_polygons_x[level] += self.merge_overlap(self.polygons_x[level])

            self.polygons_y[level] = self.extract_polygons(ellist, Npoly=64, axis="yz")
            self.merged_polygons_y[level] += self.merge_overlap(self.polygons_y[level])

            self.polygons_z[level] = self.extract_polygons(ellist, Npoly=64, axis="zx")
            self.merged_polygons_z[level] += self.merge_overlap(self.polygons_z[level])

    def extract_hull(self, ellipse, Npoly=64, axis='xy'):
        xyz = ellipse.getContour(N=Npoly)

        if axis == "xy":
            x = np.reshape(xyz[0], (Npoly, Npoly))
            y = np.reshape(xyz[1], (Npoly, Npoly))

        elif axis == "yz":
            x = np.reshape(xyz[1], (Npoly, Npoly))
            y = np.reshape(xyz[2], (Npoly, Npoly))

        elif axis == "zx":
            x = np.reshape(xyz[2], (Npoly, Npoly))
            y = np.reshape(xyz[0], (Npoly, Npoly))

        points = np.reshape(np.ravel([np.ravel(x), np.ravel(y)]), (len(np.ravel(x)), 2), order='F')
        return ConvexHull(points)

    def extract_polygons(self, ellist, axis, Npoly=64):
        polygons = []

        for ellipse in ellist:
            hull = self.extract_hull(ellipse, Npoly, axis)
            polygons.append(shp.Polygon(hull.points[hull.vertices]))

        return polygons

    def merge_overlap(self, polygons):

        area = np.mean([p.area for p in polygons])
        min_separation = np.sqrt(2 * np.log(2)) * np.sqrt(area / np.pi)

        # compute center to center distance matrix
        # print("\nminsep", min_separation)
        lst = pu.coordMatrix(polygons)
        distances = pu.sepDistance(lst)

        # build associated network
        matrix = distances < min_separation
        # print("\ndistance", distances)
        np.fill_diagonal(matrix, 0)
        # print("\ndistance ok", matrix)

        new_polygons = polygons[:]
        while np.any(matrix == True):
            G = nx.from_numpy_array(matrix)
            # G = nx.from_numpy_matrix(matrix)

            maximal_cliques = sorted(nx.find_cliques(G), key=len, reverse=True)

            # print("clique", len(maximal_cliques))
            # if len(maximal_cliques) == 0:
            #    break

            new_polygons = []
            for clique in maximal_cliques:
                poly_clique = [polygons[node] for node in clique]
                new_polygons.append(unary_union(poly_clique))

            lst = pu.coordMatrix(new_polygons)
            distances = pu.sepDistance(lst)
            matrix = distances < min_separation
            np.fill_diagonal(matrix, 0)

            polygons = new_polygons[:]

        # [new_polygons.append(polygons[node]) for node in G.nodes ]

        return new_polygons


    def save_catalogs(self, file_name, save_path='./'):
        def _apply(axis, axis_name):
            for level, polygons in axis:
                full_path = save_path + file_name + "2D_level" + str(level) + "_" + axis_name
                data = {
                    "x": [],
                    "y": []
                }

                for polygon in polygons:
                    X, Y = polygon.exterior.coords.xy
                    data["x"].append(X.tolist())
                    data["y"].append(Y.tolist())

                pd_data = pd.DataFrame(data)
                pd_data.to_csv(full_path + ".csv", index=False)

        _apply(self.merged_polygons_x.items(), "x")
        _apply(self.merged_polygons_y.items(), "y")
        _apply(self.merged_polygons_z.items(), "z")

    def plot(self, colors={0:"blue", 1:"red", 2:"green"}, merged=True):
        import matplotlib.pyplot as plt

        def _plot(merged, ax, axis):
            if merged:
                if axis == "xy":
                    poly_dic = self.merged_polygons_z
                elif axis == "yz":
                    poly_dic = self.merged_polygons_x
                elif axis == "zx":
                    poly_dic = self.merged_polygons_y
            else:
                if axis == "xy":
                    poly_dic = self.polygons_z
                elif axis == "yz":
                    poly_dic = self.polygons_x
                elif axis == "zx":
                    poly_dic = self.polygons_y

            for level, polygons in poly_dic.items():
                for polygon in polygons:
                    X, Y = polygon.exterior.coords.xy
                    ax.plot(X, Y, color=colors[level])

        fig, axs = plt.subplots(ncols=3, num=f"Merged:{merged}", figsize=(9, 3))

        axs[0].set_xlabel('$X$')
        axs[0].set_ylabel('$Y$')

        _plot(merged, axs[0], axis="xy")

        axs[1].set_xlabel('$Z$')
        axs[1].set_ylabel('$X$')

        _plot(merged, axs[1], axis="zx")

        axs[2].set_xlabel('$Y$')
        axs[2].set_ylabel('$Z$')

        _plot(merged, axs[2], axis="yz")

        plt.draw()

    """
    def isolating_graph(self, G):
        degrees = sorted(G.degree, key = lambda x: x[1], reverse = True)
        #print(G.degree)
        node, degree = degrees[0]
        
        while degree != 0:
            G.remove_node(node)
            degrees = sorted(G.degree, key = lambda x: x[1], reverse = True)
            node, degree = degrees[0]
    """

    
    def countChildren(self, df, level):
        N_3D = []
        N_2D = []

        idxs = (df["level"] == level)
        
        for sidx in np.unique(df["structure"]):
            
            polys =  df["polygons"][(df["structure"] == sidx) & (df["level"] == level)]
            polylist = polys.to_list()
            
            #if len(polylist) != 0:
                # polygons fusionnés
            merged = self.merge_overlap( polylist )
                
            N_3D.append(len(polys))
            N_2D.append(len(merged))

        return N_3D, N_2D

    def get_children_numbers(self, fragmentation_rate, scaling_ratios, overlap, Nobj):
        PHIS, R, OVER = np.meshgrid(fragmentation_rate, scaling_ratios, overlap)

        shape = [
            len(fragmentation_rate),
            len(scaling_ratios),
            len(overlap),
            Nobj,
            2
                ]
        
        print(shape)
        
        N_3Ds = np.ones(shape=shape)
        N_2Ds = np.ones(shape=shape)

        for phi3D, ratio, over in zip(np.ravel(PHIS), np.ravel(R), np.ravel(OVER)):
            df = self.open_file(phi3D, ratio, over)

            ells = [Ellipsoid(df["xo"][i], df["yo"][i], df["zo"][i], 
                              df["a"][i], df["b"][i], df["c"][i], 
                              rotation = df["rot"][i], level=df["level"][i])
                            for i in range(len(df))]
        
            df['ellispoid'] = ells
            polygons = self.extract_polygons(ells)
            df['polygons'] = polygons

            idx1 = np.argwhere(fragmentation_rate == phi3D)[0][0]
            idx2 = np.argwhere(scaling_ratios == ratio)[0][0]
            idx3 = np.argwhere(overlap == over)[0][0]
            
            for level in range(shape[-1]):
                N_3D, N_2D = self.countChildren(df, level)

                # numbers
                N_3Ds[idx1, idx2, idx3, :, level] = N_3D
                N_2Ds[idx1, idx2, idx3, :, level] = N_2D

        return N_2Ds, N_3Ds

    """
    def _open_file(self):
        #path = "/Users/thomaben/Documents/GitHub/EllispoidsPopulation/"        
        PHIS, R, OVER = np.meshgrid(fragmentation_rate, scaling_ratios, overlap)
        
        shape = list(PHIS.shape)
        shape2 = list(PHIS.shape)
        shape3 = list(PHIS.shape)
        shape.append(1000)
        shape2.append(8000)
        shape3.append(28000)
        
        N_3Ds = np.ones(shape=shape)
        N_2Ds = np.ones(shape=shape)
        
        alpha_include_2Ds = np.zeros(shape=shape2)
        alpha_covering_2Ds = np.zeros(shape=shape3)
        alpha_covering_post2Ds = np.zeros(shape=shape3)
    
        size_3Ds = np.zeros(shape=shape2)
        size_prefilts = np.zeros(shape=shape2)
        size_postfilts = np.zeros(shape=shape2)
        
        print(os.listdir(self.save_path))
    
        for phi3D, ratio, over in zip(np.ravel(PHIS), np.ravel(R), np.ravel(OVER)):
            #print("phi3D", phi3D, "r", ratio, "over", over)
            name = "phi3D_" + str(phi3D).replace('.', 'p') + "r_" + str(ratio).replace('.', 'p') + "over_" + str(over).replace('.', 'p')
            print("opening", self.save_path + name + ".csv")
            print(name + ".csv" in os.listdir(self.save_path))
            try:
                df = pd.read_csv(self.save_path + name + ".csv")
                print("success")
            except:
                df = None
    
            if df is not None:
                try:
                    df["rot"] = df["rot"].apply(reshape_func)
                except:
                    pass
    
                ells = [Ellipsoid(df["xo"][i], df["yo"][i], df["zo"][i], 
                          df["a"][i], df["b"][i], df["c"][i], 
                          rotation = df["rot"][i], level=df["level"][i])
                        for i in range(len(df))]
    
                df['ellispoid'] = ells
    
                polygons = extract_polygons(ells)
                df['polygons'] = polygons
    
                N_3D = []
                N_2D = []
                
                size_3D = []
                size_prefilt = []
                size_postfilt = []
                
                alpha_include_2D = []
                alpha_covering_2D = []
                alpha_covering_post2D = []
                
                idxs = (df["level"] == 1)
                child_area = np.mean( [poly.area for poly in df['polygons'][idxs]])
                
                size_3D = (df["a"][idxs]*df["b"][idxs]*df["c"][idxs])**(1/3)
                size_3D = size_3D.tolist()
    
                for sidx in np.unique(df["structure"]):
                    polys =  df["polygons"][(df["structure"] == sidx) & (df["level"] == 1)]
                    #print(polys.to_list())
                    #print(len(polys.to_list()))
                    
                    polylist = polys.to_list()
                    if len(polylist) != 0:
                        
                        # aire de l'intersection entre les enfants normalisé sur l'air typique
                        if len(polylist) > 1:
                            for comb in itertools.combinations(np.arange(0, len(polylist), 1), 2):   
                                intersct = polylist[comb[0]].intersection(polylist[comb[1]])
                                alpha_covering_2D.append(intersct.area / child_area)
                        
                        # aire de l'enfant incluse dans le parent
                        parent = df["polygons"][(df["structure"] == sidx) & (df["level"] == 0)][sidx-1]
    
                        for poly in polys:
                            inters = poly.intersection(parent)
                            alpha_include_2D.append( inters.area / poly.area )
                            
                            if inters.area / poly.area == 0:
                                print("zero")
                            
                            
                            size_prefilt.append(np.sqrt(poly.area/np.pi))
                        
                        # polygons fusionnés
                        merged = merge_overlap( polylist )
                        # print(len(polylist), len(merged))
                        
                        if len(merged) > 1:
                            for comb in itertools.combinations(np.arange(0, len(merged), 1), 2):                            
                                intersct = merged[comb[0]].intersection(merged[comb[1]])
                                alpha_covering_post2D.append(intersct.area)
                        
                        for poly in merged:
                            size_postfilt.append(np.sqrt(poly.area/np.pi))
                        
                    else:
                        merged = [1]
                        #alpha_covering_2D.append(np.nan)
                        #alpha_covering_post2D.append(np.nan)
    
                    N_3D.append(len(polys))
                    N_2D.append(len(merged))
                    #print(len(alpha_covering_2D))
                    
                    #print(np.sum(N_2D))
    
                idx1 = np.argwhere(fragmentation_rate == phi3D)[0][0]
                idx2 = np.argwhere(scaling_ratios == ratio)[0][0]
                idx3 = np.argwhere(overlap == over)[0][0]
                
                # numbers
                N_3Ds[idx2, idx1, idx3, :] = N_3D
                N_2Ds[idx2, idx1, idx3, :] = N_2D
                
                # covers
                alpha_covering_2Ds[idx2, idx1, idx3, :len(alpha_covering_2D)] = alpha_covering_2D
                alpha_covering_post2Ds[idx2, idx1, idx3, :len(alpha_covering_post2D)] = alpha_covering_post2D
    
                alpha_include_2Ds[idx2, idx1, idx3, :len(alpha_include_2D)] = alpha_include_2D
    
                # sizes
                #parents = df["polygons"][df["level"] == 0]
                size_3Ds[idx2, idx1, idx3, :len(size_3D)] = size_3D #[np.sqrt(parent.area/np.pi) for parent in parents]
    
                size_prefilts[idx2, idx1, idx3, :len(size_prefilt)] = size_prefilt
                size_postfilts[idx2, idx1, idx3, :len(size_postfilt)] = size_postfilt
                
        Numbers = [N_3Ds, N_2Ds]
        Coverage = [alpha_covering_2Ds, alpha_covering_post2Ds, alpha_include_2Ds]
        Sizes = [size_3Ds, size_prefilts, size_postfilts]
    
        return Numbers, Coverage, Sizes
    """

class Mapper:
    def __init__(self, ellipses_data, metadata = None,
                 xlim = (-15, 15),
                 ylim = (-15, 15),
                 zlim = (-15, 15), shape = 256):

        self.data = ellipses_data
        self.metadata = metadata

        self.x_axis = np.linspace(xlim[0], xlim[1], shape)
        self.y_axis = np.linspace(ylim[0], ylim[1], shape)
        self.z_axis = np.linspace(zlim[0], zlim[1], shape)
        self.shape = shape

    def build_map(self):
        X, Y, Z = np.meshgrid(self.x_axis, self.y_axis, self.z_axis)

        number_objects = len(self.data["xo"])
        self.total_cube = np.empty(shape=X.shape)

        print("size of data", getsizeof(self.data))
        print("size of X grid", getsizeof(X))
        print("size of final cube", getsizeof(self.total_cube))

        for i in tqdm(range(number_objects)):
            mu_xyz = (self.data["xo"][i],
                      self.data["yo"][i],
                      self.data["zo"][i])

            sigma_xyz = (self.data["a"][i]**2,
                         self.data["b"][i]**2,
                         self.data["c"][i]**2)

            self.total_cube += statfrag.set_gaussian_cube(
                grid_coords=(X, Y, Z),
                mu_xyz=mu_xyz,
                sigma_xyz=sigma_xyz,
                rotation = self.data["rot"][i])


        """
        mu_xyz = (self.data["xo"].to_numpy(),
                  self.data["yo"].to_numpy(),
                  self.data["zo"].to_numpy())
        sigma_xyz = (self.data["a"].to_numpy(),
                     self.data["b"].to_numpy(),
                     self.data["c"].to_numpy())

        print(mu_xyz)

        cubes = statfrag.set_gaussian_cube(
            grid_coords=(X[:, :, :, np.newaxis],
                         Y[:, :, :, np.newaxis],
                         Z[:, :, :, np.newaxis]),
            mu_xyz=mu_xyz,
            sigma_xyz=sigma_xyz)
        
        """

        """
        chunk = 128
        number_objects = len(self.data["xo"])
        shape_cubes = tuple(list(X.shape) + [number_objects])
        cubes = np.empty(shape=shape_cubes)
        for i in range(0, number_objects % chunk):
            max_idx = min(i * (chunk + 1), number_objects)

            mu_xyz = (self.data["xo"][ i*chunk:max_idx ].to_numpy(),
                      self.data["yo"][ i*chunk:max_idx ].to_numpy(),
                      self.data["zo"][ i*chunk:max_idx ].to_numpy())
            sigma_xyz = (self.data["a"][ i*chunk:max_idx ].to_numpy(),
                         self.data["b"][ i*chunk:max_idx ].to_numpy(),
                         self.data["c"][ i*chunk:max_idx ].to_numpy())

            print(self.data["xo"])

            cubes[:, :, :, i*chunk:max_idx] = statfrag.set_gaussian_cube(
                grid_coords=(X[:, :, :, np.newaxis],
                             Y[:, :, :, np.newaxis],
                             Z[:, :, :, np.newaxis]),
                mu_xyz=mu_xyz,
                sigma_xyz=sigma_xyz)
        """

    def project_cube(self):
        from scipy.integrate import trapezoid
        self.projected_cube_x = trapezoid(self.total_cube, x=self.x_axis, axis=2)
        self.projected_cube_y = trapezoid(self.total_cube, x=self.y_axis, axis=1)
        self.projected_cube_z = trapezoid(self.total_cube, x=self.z_axis, axis=0)

    def plot(self):
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(ncols=3, figsize=(9, 3))

        axs[0].set_xlabel('$X$')
        axs[0].set_ylabel('$Y$')

        X, Y = np.meshgrid(self.x_axis, self.y_axis)
        axs[0].pcolormesh(X, Y, self.projected_cube_z)

        axs[1].set_xlabel('$Z$')
        axs[1].set_ylabel('$X$')

        X, Z = np.meshgrid(self.x_axis, self.z_axis)
        axs[1].pcolormesh(Z, X, self.projected_cube_y)

        axs[2].set_xlabel('$Y$')
        axs[2].set_ylabel('$Z$')

        Y, Z = np.meshgrid(self.y_axis, self.z_axis)
        axs[2].pcolormesh(Y, Z, self.projected_cube_x)

        plt.draw()

    def save_map(self, file_name, save_path="./"):
        from astropy.io import fits

        def set_header(hdu, axis1, axis2):
            hdu.header["CRVAL1"] = axis1[0]
            hdu.header["CRPIX1"] = 0
            hdu.header["CDELT1"] = axis1[1] - axis1[0]

            hdu.header["CRVAL2"] = axis2[0]
            hdu.header["CRPIX2"] = 0
            hdu.header["CDELT2"] = axis2[1] - axis2[0]

            hdu.header["CTYPE1"] = "LINEAR  "
            hdu.header["CTYPE2"] = "LINEAR  "

            hdu.header["CUNIT1"] = "deg     "
            hdu.header["CUNIT2"] = "deg     "

        hdu = fits.PrimaryHDU(data = self.projected_cube_x)
        set_header(hdu, self.y_axis, self.z_axis)
        hdu.writeto(save_path + file_name + '_x.fits', overwrite=True)

        hdu = fits.PrimaryHDU(data=self.projected_cube_y)
        set_header(hdu, self.z_axis, self.x_axis)
        hdu.writeto(save_path + file_name + '_y.fits', overwrite=True)

        hdu = fits.PrimaryHDU(data=self.projected_cube_z)
        set_header(hdu, self.x_axis, self.y_axis)
        hdu.writeto(save_path + file_name + '_z.fits', overwrite=True)



def open_population(file_name, format="csv", save_path='./'):
    if format == "csv":
        return _open_population_csv(file_name, save_path=save_path)
    elif format == "hdf5":
        pass
    else:
        raise ("format arg not recognised. Please enter one of the following format as a string:\n"
               "\t csv\n"
               "\t hdf5")

def _open_population_csv(file_name, save_path="./"):
    # name = "phi3D_" + str(phi3D).replace('.', 'p') + "r_" + str(ratio).replace('.', 'p') + "over_" + str(over).replace('.', 'p')
    print("opening", save_path + file_name + ".csv")
    df = pd.read_csv(save_path + file_name + ".csv")

    df["rot"] = df["rot"].apply(reshape_func)

    return df

def reshape_func(row):
    text = row.replace("\n", " ").replace('[', " ").replace(']', " ")
    return np.reshape(np.fromstring(text, sep=' '), (3, 3))
                
def phi(N0, N1, r):
    return np.log(N1/N0) / np.log(r)

def main_create(wdir):
    """
    Input models
    """
    CoordModel = statfrag.CoordinatesPDF(name='uniform')  # PDF of coordinates selection
    # EllipseModel = statfrag.CoordinatesPDF(name='uniform') #PDF of ellipsoid parameters (a, b, c)
    # EulerAngleModel = statfrag.CoordinatesPDF(name='uniform') #PDF of ellipsoid orientation angles

    DiscreteModel = statfrag.DiscretePDF(name="binary")  # PDF of the number of fragments

    Nvertices = 32
    Nobj = 10

    """
    Setup an initial population
    """
    a = np.random.normal(1, 0.1, size=Nobj)
    b = np.random.uniform(low=0.7, high=1, size=Nobj) * a
    c = np.random.uniform(low=1, high=1.3, size=Nobj) * a

    xo, yo, zo = np.random.uniform(low=-10, high=10, size=(3, Nobj))
    prec, nut, gir = np.random.uniform(0, 90, size=(3, Nobj))

    initial_parents = []
    for i in range(Nobj):
        initial_parents.append(Ellipsoid(xo[i], yo[i], zo[i],
                                         a[i], b[i], c[i],
                                         rotation=(prec[i], nut[i], gir[i]),
                                         level=0, N=Nvertices, sidx=i))

    """
    Generates the children with corresponding fragmentation law
    """
    # path = "./"
    fragmentation_rate = np.repeat(1.1, 2)
    scaling_ratios = np.repeat(2, 2)

    setup = Generator(initial_parents,
                      fragmentation_rates=fragmentation_rate,
                      scaling_ratios=scaling_ratios,
                      overlap=0.9,
                      DiscreteModel=DiscreteModel,
                      CoordModel=CoordModel)

    setup.buildPopulation_3D()

    colors = {0: "blue", 1: "red", 2: "green"}
    setup.plot(colors=colors, Nvertices=16)

    setup.save_population(format="csv",
                          file_name="test_save",
                          save_path=wdir)

def main_open(wdir):
    import matplotlib.pyplot as plt
    data = open_population("test_save", save_path=wdir)

    projection = Projector(data)
    projection.project_population()

    projection.plot(merged=False)
    projection.plot(merged=True)

    mapper = Mapper(data)
    mapper.build_map()
    mapper.project_cube()
    mapper.plot()

    plt.show()
    projection.save_catalogs("test_save", save_path=wdir)
    mapper.save_map("test_map", save_path=wdir)

if __name__=="__main__":
    #main_create("../synthetic_data/test_files/")
    main_open(wdir = "../synthetic_data/test_files/")