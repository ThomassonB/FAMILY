import numpy as np

import shapely.geometry as shp
from scipy.spatial import ConvexHull, convex_hull_plot_2d

from . import polygons_utility as pu
import networkx as nx
from shapely.ops import unary_union
import pandas as pd

from tqdm import tqdm
import copy, os

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
    
    def children(self, N, size, sep, limit = 10_000, impen=1e-3, **kwargs):
        self.ellchild = []
        X = np.array([])
        Y = np.array([])
        Z = np.array([])

        # create a pool of coordinates        
        a = np.random.normal(1*size, 0.1*size, size = limit)
        b = np.random.uniform(low=0.7, high=1, size=limit) * a
        c = np.random.uniform(low=1, high=1.3, size=limit) * a
        
        prec, nut, gir = np.random.uniform(0, 90, size=(3, limit))

        xo = np.random.uniform(low=-self.a, high=self.a, size=limit)
        yo = np.random.uniform(low=-self.b, high=self.b, size=limit)
        zo = np.random.uniform(low=-self.c, high=self.c, size=limit)
        
        ro = xo ** 2 + yo ** 2 + zo ** 2
        
        xyz = rotateEllipsoid([xo,yo,zo], self.rotation)
        xo = xyz[0] + self.centroid.xo
        yo = xyz[1] + self.centroid.yo
        zo = xyz[2] + self.centroid.zo

        count = 0
        
        while (len(self.ellchild) < N and count < limit) or len(self.ellchild)==0:
            
            if count >= limit:
                count = 0
                xo = np.random.uniform(low=-self.a, high=self.a, size=limit)
                yo = np.random.uniform(low=-self.b, high=self.b, size=limit)
                zo = np.random.uniform(low=-self.c, high=self.c, size=limit)

                ro = xo ** 2 + yo ** 2 + zo ** 2

                xyz = rotateEllipsoid([xo,yo,zo], self.rotation)
                xo = xyz[0] + self.centroid.xo
                yo = xyz[1] + self.centroid.yo
                zo = xyz[2] + self.centroid.zo
                
            # draw coordinate
            r = ro[count]
            
            # coordinates check 1) inside parent 2) hard sphere 3) exclusion area rmin (help packing)
            if N == 1 or (count > limit * 0.95 and len(self.ellchild) == 0):
                rmin = 0
                xo[count:] = 0
                yo[count:] = 0
                zo[count:] = 0
                
                ro[count:] = xo[count:] ** 2 + yo[count:] ** 2 + zo[count:] ** 2
        
                xyz = rotateEllipsoid([xo[count:], yo[count:], zo[count:]], self.rotation)
                xo[count:] = xyz[0] + self.centroid.xo
                yo[count:] = xyz[1] + self.centroid.yo
                zo[count:] = xyz[2] + self.centroid.zo
                
            else:
                rmin = sep / 4
                    
            if r >= rmin:
                ellchild = Ellipsoid(xo[count], yo[count], zo[count], 
                                     a[count], b[count], c[count],
                                     rotation=(prec[count], nut[count], gir[count]),
                                     level = self.level + 1, sidx = self.sidx)
                
                hard = [not ellchild.within(other, minimum = impen) for other in self.ellchild]
                
                #self.within(ellchild, **kwargs, verbose=True)
                
                if self.within(ellchild, **kwargs) and np.all(hard):
                    X = np.append(X, xo[count])
                    Y = np.append(Y, yo[count])
                    Z = np.append(Z, zo[count])
                    self.ellchild.append(ellchild)
                    
            count += 1
                        
    def genealogy(self, ell, Nl, sl, sep, **kwargs):
        self.children(N = Nl[l], size = sl[l], sep = sl[l])
        self.genealogy(self.ellchild, N = Nl[l], size = sl[l], sep = sl[l], **kwargs)

class Generator:
    def __init__(self, initial_parents, save_path):
        self.initial_parents = initial_parents
        self.save_path = save_path

    def save_population(self, ells, phi3D, ratio, overlap):
        
        data = {
        "xo":[],
        "yo":[],
        "zo":[],
        "a":[],
        "b":[],
        "c":[],
        "rot":[],
        "level":[],
        "structure":[],
        }
    
        for ell in ells:
            data["xo"].append(    ell.centroid.xo)
            data["yo"].append(    ell.centroid.yo)
            data["zo"].append(    ell.centroid.zo)
            data["a"].append(     ell.a)
            data["b"].append(     ell.b)
            data["c"].append(     ell.c)
            data["rot"].append(   ell.rotation)
            data["level"].append( ell.level)
            data["structure"].append( ell.sidx)
        
        pd_data = pd.DataFrame(data)
        
        phi3D = np.round(phi3D, decimals=2)
        ratio = np.round(ratio, decimals=2)
        overlap = np.round(overlap, decimals=2)
        name = "phi3D_" + str(phi3D).replace('.', 'p') + "r_" + str(ratio).replace('.', 'p') + "over_" + str(overlap).replace('.', 'p')
        
        pd_data.to_csv(self.save_path + name + ".csv", index=False) 

    def selectFragmentNumber(self, N):
        import bisect
    
        p = N - int(N)
        p = [1 - p, p]
        n = [int(N), int(N) + 1]
    
        N_discrete = n[bisect.bisect_left(np.cumsum(p), np.random.rand())]
    
        N_discrete = max(N_discrete, 1)
        return int(N_discrete)

    def mapPopulation_3D(self, initial_parents, fragmentation_rates, scaling_ratios, overlaps, Npoly = 64):
        PHIS, R, OVER = np.meshgrid(fragmentation_rates, scaling_ratios, overlaps)
        
        for phi3D, ratio, over in zip(PHIS.flatten(), R.flatten(), OVER.flatten()):
            N = ratio ** phi3D
            ellc = copy.deepcopy(initial_parents)
            size = 1/ratio
            sep = size**2
            
            children = []
            for ell in tqdm(ellc):
                
                n = self.selectFragmentNumber(N)
                ell.children(N = n, size = size, sep = sep, minimum = over)
                children += ell.ellchild
                
            total = ellc + children
            self.save_population(total, phi3D, ratio, over)

    def buildPopulation_3D(self, initial_parents, fragmentation_rates, scaling_ratios, overlaps, Npoly = 64):
        #PHIS, R, OVER = np.meshgrid(fragmentation_rate, scaling_ratios, overlap)

        ellipses = copy.deepcopy(initial_parents)
        
        for phi3D, ratio, overlap in zip(fragmentation_rates, scaling_ratios, overlaps):
            number = ratio ** phi3D
            size = 1/ratios
            sep = size**2
                        
            children = []
            for parent in tqdm(initial_parents):
                n = selectFragmentNumber(number)
                parent.children(N = n, size = size, sep = sep, minimum = overlap)
                children += ell.ellchild
                    
            initial_parents = children
            ellipses += children
        
        self.save_population(ellipses, 
                             np.mean(fragmentation_rates), 
                             np.mean(scaling_ratios), 
                             np.mean(overlaps)
                            )

    def projectPopulation3D(self):

        df = open_file()

        merged_polygons = []
        for level in df["level"].unique():
            sub_df = df['level'] == level
            
            ellist = [Ellipsoid(df["xo"][i], df["yo"][i], df["zo"][i], 
                                df["a"][i], df["b"][i], df["c"][i], 
                                rotation = df["rot"][i], level=df["level"][i])
                    for i in df[sub_df].index]
            
            polygons = self.extract_polygons(ellist, Npoly = 64)
            merged_polygons += self.merge_overlap(polygons)
            
        return merged_polygons
        #saveCatalog2D(merged_polygons, df.header)   

    def saveCatalog2D(self):
        pass

    def isolating_graph(self, G):
        degrees = sorted(G.degree, key = lambda x: x[1], reverse = True)
        #print(G.degree)
        node, degree = degrees[0]
        
        while degree != 0:
            G.remove_node(node)
            degrees = sorted(G.degree, key = lambda x: x[1], reverse = True)
            node, degree = degrees[0]
    
    def extract_hull(self, ellipse, Npoly = 64):
        xyz = ellipse.getContour(Npoly)
        x = np.reshape(xyz[2], (Npoly, Npoly))
        y = np.reshape(xyz[0], (Npoly, Npoly))
    
        points = np.reshape(np.ravel([np.ravel(x), np.ravel(y)]), (len(np.ravel(x)), 2), order='F')
        return ConvexHull(points)
    
    def extract_polygons(self, ellist, Npoly = 64):
        polygons = []
        
        for ellipse in ellist:
            hull = self.extract_hull(ellipse, Npoly)
            polygons.append( shp.Polygon( hull.points[hull.vertices] ) )
            
        return polygons
    
    def merge_overlap(self, polygons):
        
        area = np.mean([p.area for p in polygons])
        min_separation = np.sqrt(2*np.log(2)) * np.sqrt(area / np.pi)
    
        # compute center to center distance matrix
        #print("\nminsep", min_separation)
        lst = pu.coordMatrix(polygons)
        distances = pu.sepDistance(lst)
    
        # build associated network
        matrix = distances < min_separation
        #print("\ndistance", distances)
        np.fill_diagonal(matrix, 0)
        #print("\ndistance ok", matrix)
    
        new_polygons = polygons[:]
        while np.any(matrix == True):
            G = nx.from_numpy_array(matrix)
            #G = nx.from_numpy_matrix(matrix)
    
            maximal_cliques = sorted(nx.find_cliques(G), key=len, reverse=True)
    
            #print("clique", len(maximal_cliques))
            #if len(maximal_cliques) == 0:
            #    break
                
            new_polygons = []
            for clique in maximal_cliques:
                poly_clique = [polygons[node] for node in clique]
                new_polygons.append( unary_union(poly_clique) )
    
            lst = pu.coordMatrix(new_polygons)
            distances = pu.sepDistance(lst)
            matrix = distances < min_separation
            np.fill_diagonal(matrix, 0)
            
            polygons = new_polygons[:]
        
        #[new_polygons.append(polygons[node]) for node in G.nodes ]
        
        return new_polygons

    def open_file(self, phi3D, ratio, over):
        name = "phi3D_" + str(phi3D).replace('.', 'p') + "r_" + str(ratio).replace('.', 'p') + "over_" + str(over).replace('.', 'p')
        print("opening", self.save_path + name + ".csv")
        print("file in path:", name + ".csv" in os.listdir(self.save_path))
        try:
            df = pd.read_csv(self.save_path + name + ".csv")
            print("success")
        except:
            df = None
            print("failure")

        if df is not None:
            try:
                df["rot"] = df["rot"].apply(reshape_func)
            except:
                pass
        return df
    
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

def reshape_func(row):
    text = row.replace("\n", " ").replace('[', " ").replace(']', " ")
    return np.reshape(np.fromstring(text, sep=' '), (3, 3))
                
def phi(N0, N1, r):
    return np.log(N1/N0) / np.log(r)

if __name__=="__main__":
    pass