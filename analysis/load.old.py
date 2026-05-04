import os
import tomllib

import numpy as np
import pandas as pd 
import shapely.geometry.polygon as shp

_init_default_dict = {
    "object_type":1,
    "x_coord_name":"X",
    "y_coord_name":"Y",
    "major_axis_name":"A_FWHM",
    "minor_axis_name":"B_FWHM",
    "position_angle_name":"Theta",
    "x_coord_conversion_to_deg":1,
    "y_coord_conversion_to_deg":1,
    "major_conversion_to_hwhm":0.5, #half width at half maximum
    "minor_conversion_to_hwhm":0.5, #half width at half maximum
    "angle_conversion_to_radian":1,
    "offset_angle": 0,
    "beam":1,
    "wavelength":1,
    "source_extent":1,
    "cloud_distance":1,
    "catalog_path":"./",
    "image_path":"./",
    "extraction_window":-1,
    "class_name":"Class",
}

def get_keywords():
    return set(_init_default_dict.keys())


def add_new(init_name, path, **kwargs):
    if init_name[:-5] != "_init":
        init_name += "_init"

    init_dict = _init_default_dict.copy()
    for key, value in kwargs.items():
        init_dict[key] = value


    len_key = len(max(init_dict, key=len))
    len_values = len(max([str(value) for value in init_dict.values()], key=len))
    line = '{0:<%d}\t{1:<%d}\t{2}\n' % (len_key, len_values)

    with open(path + init_name, "w+") as init_file:
        init_file.write(
            f"{line.format("keyword", "value", "description")}"
            f"_____________________________________________________\n"
            f"{line.format("catalog_path", init_dict["catalog_path"],   "Path of catalog")}"
            f"{line.format("image_path", init_dict["image_path"],       "Path of image")}"
            f"{line.format("object_type", init_dict["object_type"],     "Objects type of catalog (yso is 0, ellipse is 1, polygon is 2)")}"
            f"{line.format("beam", init_dict["beam"],                   "Beam used for the catalog (arcsec)")}"
            f"{line.format("wavelength", init_dict["wavelength"],       "Wavelength of observation (microm)")}"
            f"{line.format("cloud_distance", init_dict["cloud_distance"], "Distance (pc)")}"
            f"{line.format("x_coord_name", init_dict["x_coord_name"], "Name of X coordinates column in the catalog")}"
            f"{line.format("y_coord_name", init_dict["y_coord_name"], "Name of Y coordinates column in the catalog")}"
            f"{line.format("x_coord_conversion_to_deg", init_dict["x_coord_conversion_to_deg"], "Conversion factor to have degrees")}"
            f"{line.format("y_coord_conversion_to_deg", init_dict["y_coord_conversion_to_deg"], "Conversion factor to have degrees")}"
        )

        if init_dict["object_type"] == 0:
            init_file.write(f"{line.format("class_name", init_dict["class_name"], "Name of YSO class column")}")

        elif init_dict["object_type"] == 1:
            init_file.write(
                f"{line.format("major_axis_name", init_dict["major_axis_name"], "Name of major axis column")}"
                f"{line.format("minor_axis_name", init_dict["minor_axis_name"], "Name of minor axis column")}"
                f"{line.format("major_conversion_to_hwhm", init_dict["major_conversion_to_hwhm"], "Conversion factor to have degrees and half width at half maximum")}"
                f"{line.format("minor_conversion_to_hwhm", init_dict["minor_conversion_to_hwhm"], "Conversion factor to have degrees and half width at half maximum")}"
                f"{line.format("source_extent", init_dict["source_extent"], "Extent of sources (default is the FWHM)")}"
                f"{line.format("position_angle_name", init_dict["position_angle_name"], "Name of position angle column")}"
                f"{line.format("angle_conversion_to_radian", init_dict["angle_conversion_to_radian"], "Conversion factor to have radian")}"
                f"{line.format("offset_angle", init_dict["offset_angle"], "Angle offset (rad)")}"
                            )

    print(f"file {init_name} created at:\n"
          f"\t{path}")

def read_init(file):
    metadata = {}
    with open(file, 'r') as f:
        lines = f.readlines()[2:]
        for line in lines:
            key, value, *_ = line.split()
            try:
                metadata[key] = float(value)
            except:
                metadata[key] = value
    return metadata

def load_data(file, reader):
    metadata = read_init(file)
    strings_ref = ["_X", "_Y", "_A", "_B", "_Theta"]

    data_frame = reader(metadata["catalog_path"])
    if metadata["object_type"] == 0:
        data_frame[strings_ref[0]] = pd.Series([float(i) for i in data_frame[metadata["x_coord_name"]]], index=data_frame.index)
        data_frame[strings_ref[1]] = pd.Series([float(i) for i in data_frame[metadata["y_coord_name"]]], index=data_frame.index)

        #circular source of beam size (in degree)
        data_frame[strings_ref[2]] = pd.Series(np.ones_like(metadata["x_coord_name"]) * metadata["beam"] / 3600, index=data_frame.index)
        data_frame[strings_ref[3]] = pd.Series(np.ones_like(metadata["x_coord_name"]) * metadata["beam"] / 3600, index=data_frame.index)
        data_frame[strings_ref[4]] = pd.Series(np.zeros_like(metadata["x_coord_name"]), index=data_frame.index)

        # set the size of object (in arcsec)
        data_frame["_R"] = data_frame[strings_ref[2]] * 3600

        # get classes in labelled
        # data_frame["_Class"] = data_frame[metadata["class_name"]]

        # keep only elements inside window
        #points = [shp.Point(float(data_frame[strings_ref[0]][i]), float(data_frame[strings_ref[1]][i]))
        #          for i in range(len(data_frame.index))]

        #data_frame['iswithin'] = [point.within(window) for point in points]

        #data_frame = data_frame[data_frame.Flag0 == True].reset_index()
        #data_frame['Mass'] = 0

    elif metadata["object_type"] == 1:
        data_frame[strings_ref[0]] = data_frame[metadata["x_coord_name"]] * metadata["x_coord_conversion_to_deg"]
        data_frame[strings_ref[1]] = data_frame[metadata["y_coord_name"]] * metadata["y_coord_conversion_to_deg"]

        data_frame[strings_ref[2]] = data_frame[metadata["major_axis_name"]] * metadata["major_conversion_to_hwhm"] * metadata["source_extent"]
        data_frame[strings_ref[3]] = data_frame[metadata["minor_axis_name"]] * metadata["minor_conversion_to_hwhm"] * metadata["source_extent"]
        data_frame[strings_ref[4]] = data_frame[metadata["position_angle_name"]] * metadata["angle_conversion_to_radian"] + metadata["offset_angle"]

        # set the size of object (in arcsec)
        data_frame["_R"] = np.sqrt(data_frame[metadata["major_axis_name"]] * data_frame[metadata["minor_axis_name"]]) * 3600  # in arcsec

    elif metadata["object_type"] == 2:
        data_frame[strings_ref[0]] = data_frame[metadata["x_coord_name"]] * metadata["x_coord_conversion_to_deg"]
        data_frame[strings_ref[1]] = data_frame[metadata["y_coord_name"]] * metadata["y_coord_conversion_to_deg"]

    dataset = {"catalog": data_frame, "strings": strings_ref}
    dataset.update(metadata)
    return dataset

def addInit(file):
    if os.path.exists(file):
        os.remove(file)
        
    f = open(file,"w+")
    
    f.write("___________________________________ \n")
    f.write("Objects (ellipses is 0, polygons is 1) \n")
    f.write("0 ; \n")
    
    f.write("___________________________________ \n")
    f.write("Names of the ellipses parameters \n")
    f.write("[X ; (deg WCS)] ; [Y ; (deg WCS)] ; [a ; (deg)] ; [b ; (deg)] ; [PA ; (rad)] \n")
    f.write("X ; Y ; A_FWHM ; B_FWHM ; Theta ; \n")
    
    f.write("___________________________________ \n")
    f.write("Eventual unit conversion \n")
    f.write("1 arcsec = %s deg ; 1 deg = %s rad \n"%(1/3600,np.pi/180)) #divide by 0.5 to have 2 sigma radius
    f.write("1 ; 1 ; 0.5 ; 0.5 ; 1 ; \n")
    
    f.write("___________________________________ \n")
    f.write("Beams used for the catalog (arcsec) ; Wavelenght of observation (microm) \n")
    f.write("0 ; 0 ; \n")
    
    f.write("___________________________________ \n")
    f.write("Extent of sources (default correspond to 2 sigma of gaussian diameter if inputted at FWHM) \n") #FWHM -> beam diameter
    f.write("%s ; \n"%(1/np.sqrt(2*np.log(2))))
    
    f.write("___________________________________ \n")
    f.write("Angle offset (rad) \n")
    f.write("%s ; \n"%(np.pi/2))
    
    f.write("___________________________________ \n")
    f.write("Distance (pc) \n")
    f.write("723 ; \n")
    
    f.write("___________________________________ \n")
    f.write("Header line ; empty lines after header (start/stop) \n")
    f.write("0 ; 0 ; 1 ; \n")
    
    f.write("___________________________________ \n")
    f.write("Window coordinates as x1 , y1 ; x2 , y2 ; ... \n")
    f.write("0 , 0 ; 1 , 1 , ... ; \n")
    
    f.write("___________________________________ \n")
    f.write("yso (0) or core (1) \n")
    f.write("1 ; \n")

    f.write("___________________________________ \n")
    f.write("YSO class column name and labels \n")
    f.write("Class ; 0/I ; II ; ")

    f.close()

def readInit(file):
    f = open(file)
    args = {}
    split_test = [";", " ; ", "; ", " ;"]

    for l, line in enumerate(f):
        words = line.split(" ; ")
        # print(words)
        if l == 0:
            args["path"] = words[0][:-1]
        elif l == 1:
            args["image"] = words[0][:-1]
        elif l == 4:
            args["obj"] = int(words[0])
        elif l == 8:
            args["strings"] = words[:-1]
        elif l == 12:
            args["conversions"] = [float(word) for word in words[:-1]]
        elif l == 15:
            args["beam"] = float(words[0])
            args["wavelength"] = float(words[1])
        elif l == 18:
            args["FWHM"] = float(words[0])
        elif l == 21:
            args["offth"] = float(words[0])
        elif l == 24:
            args["D"] = float(words[0])
        elif l == 27:
            args["header"] = [int(word) for word in words[:-1]]
        elif l == 30:
            w = []
            for tpls in words[:-1]:
                # print(tpls)
                tpl = tpls.split(" , ")
                # print(tpl)
                w.append((float(tpl[0]), float(tpl[1])))
            args["window"] = w
        elif l == 33:
            args["core"] = int(words[0])

        if l == 36:
            args["classes"] = words

    return args

def loadData(file, reader=None, **pandas_kwargs):
    if not reader:
        args = readInit(file)

        strings_ref = ["_X", "_Y", "_A", "_B", "_Theta"]

        file = args["path"]
        obj = args["obj"]
        strings = args["strings"]
        conversions = args["conversions"]
        FWHM = args["FWHM"]
        D = args["D"]
        offth = args["offth"]
        beam = args["beam"]
        wavelength = args["wavelength"]
        head = args["header"]
        window = shp.Polygon(args["window"])
        core = args["core"]

        if obj == 0:
            data = pd.read_csv(file, **pandas_kwargs)

            if core:
                data = data.drop(0)
                data = data.shift(periods=2, axis="columns")
                data = data.drop(data.columns[[0, 1]], axis=1).reset_index(drop=True)

                data[strings_ref[0]] = data[strings[0]] * conversions[0]
                data[strings_ref[1]] = data[strings[1]] * conversions[1]
                data[strings_ref[2]] = data[strings[2]] * conversions[2] * FWHM
                data[strings_ref[3]] = data[strings[3]] * conversions[3] * FWHM
                data[strings_ref[4]] = data[strings[4]] * conversions[4] + offth
                sizes = np.sqrt(data[strings_ref[2]] * data[strings_ref[3]]) * 3600  # in arcsec
                data["wave"] = wavelength

            else:
                data[strings_ref[0]] = pd.Series([float(i) for i in data[strings[0]]], index=data.index)
                data[strings_ref[1]] = pd.Series([float(i) for i in data[strings[1]]], index=data.index)
                data[strings_ref[2]] = pd.Series(np.ones_like(data[strings[0]]) * beam / 3600, index=data.index)
                data[strings_ref[3]] = pd.Series(np.ones_like(data[strings[0]]) * beam / 3600, index=data.index)
                data[strings_ref[4]] = pd.Series(np.zeros_like(data[strings[0]]), index=data.index)

                idx = np.isin(data[args["classes"][0]], args["classes"][1:])
                data = data[idx].reset_index()
                points = [shp.Point(float(data[strings_ref[0]][i]), float(data[strings_ref[1]][i])) for i in
                          range(len(data.index))]

                data['Flag0'] = [point.within(window) for point in points]

                data = data[data.Flag0 == True].reset_index()
                data['Mass'] = 0
                sizes = data[strings_ref[2]] * 3600  # in arcsec

                data["_Class"] = data[args["classes"][0]]

            data['_R'] = sizes

        elif obj == 2:
            data = pd.read_csv(file, **pandas_kwargs)

            data[strings[0]] = data[strings[0]] * conversions[0]
            data[strings[1]] = data[strings[1]] * conversions[1]
            data[strings[2]] = pd.Series(np.ones_like(data[strings[0]]) * beam / 3600, index=data.index)
            data[strings[3]] = pd.Series(np.ones_like(data[strings[0]]) * beam / 3600, index=data.index)
            data[strings[4]] = pd.Series(np.zeros_like(data[strings[0]]), index=data.index)

            data["Class"] = pd.Series(['sink'] * len(data[strings[0]]), index=data.index)

            data = data[data.Class == 'sink'].reset_index()
            points = [shp.Point(float(data[strings[0]][i]), float(data[strings[1]][i])) for i in range(len(data.index))]

            data['Flag0'] = [point.within(window) for point in points]
            data = data[data.Flag0 == True].reset_index()
            data['Mass'] = 0

            data.rename(columns={string: strings_ref[i] for i, string in enumerate(strings)}, inplace=True)

        return {"catalog": data,
                "beam": beam,
                "strings": strings_ref,
                "distance": D,
                "window": window,
                "metadata": args
                }

    else:
        return reader(file, **pandas_kwargs)


def prepareAnalyse_Dendro(file, image):
    import imageutility as iu
    def openPolygons(file):
        import pandas as pd

        metadata = dict()
        with open(file, 'r') as f:
            for l, line in enumerate(f):
                if l<7:
                    words = line.split(": ")
                    metadata[words[0]] = words[1][:-1]
                else:
                    break

        df = pd.read_csv(file, sep=";", header=7, index_col=0)

        for col in df:
            for row in df[col].index:
                if "WCS" in col:
                    df[col][row] = np.fromstring(df[col][row][1:-1], sep=',')
                else:
                    df[col][row] = np.fromstring(df[col][row][12:-2], sep=',')
            df[col] = list(df[col])

        return df, metadata

    data, metadata = openPolygons(file)

    scale1, scale2 = metadata["spatial_scale"][1:-1].split(" - ")
    scale1, scale2 = float(scale1), float(scale2)

    array, hdr = iu.OpenImage(image)

    mean_log = (np.log10(scale1) + np.log10(scale2)) / 2
    beam = abs(hdr["CDELT1"]) * 10 ** mean_log #degree
    beam = beam * 3600 #arcsec

    strings_ref = ["Poly_XWCS", "Poly_YWCS"]


    x = [0, array.shape[0], array.shape[0], 0]
    y = [0, 0, array.shape[1], array.shape[1]]
    x, y = iu.PIXtoWCS((x, y), image)
    window = shp.Polygon(np.reshape(np.ravel([x, y]), (len(x), 2), order='F'))

    X = []
    Y = []
    for xrow, yrow in zip(data[strings_ref[0]], data[strings_ref[1]]):
        X.append(np.mean(xrow))
        Y.append(np.mean(yrow))

    data["_X"] = X
    data["_Y"] = Y

    return {"catalog": data,
            "beam": beam,
            "strings": strings_ref,
            "window": window,
            "metadata": metadata
            }