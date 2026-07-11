# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 22:37:37 2023

@author: USER
"""

# Note: for ENVI format read 

# =============================================================================
# <Working Code> Just to see the Image in RGB
# Link: https://eufat.github.io/2019/02/19/hyperspectral-image-preprocessing-with-python.html
# =============================================================================
import numpy as np
from spectral import imshow, view_cube
import spectral.io.envi as envi
import matplotlib.pyplot as plt
import rasterio


# path for tiff format image
path_tiff = r"\\172.20.125.61\mtech\Major Jarmal Singh\Data_Demmin_2015_10_01\Data HSI\Smoothed_images\SpectralSubset_west_worked\SOC Prediction Image\1.by PLSR\zero_to_50_soc_pred_img.tiff"
image = rasterio.open(path_tiff)
image = image.read()
image = np.moveaxis(image, 0, -1) # moving number of bands at last
imshow(image) # SOC prediction image has only one band


# =============================================================================
# <Working Code> Checking if given coordinate (lat,long) is in raster image file (TIFF file) and 
# how to extract it with NxN window [closed]
# =============================================================================
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
import numpy as np
import pandas as pd

lat = 53.793376	
lon = 12.972641

filename = r"\\172.20.125.61\mtech\Major Jarmal Singh\Data_Demmin_2015_10_01\Data HSI\Smoothed_images\SpectralSubset_west_worked\SOC Prediction Image\1.by PLSR\zero_to_50_soc_pred_img.tiff"

with rasterio.open(filename) as rds:
    # convert coordinate to raster projection
    transformer = Transformer.from_crs("EPSG:4326", rds.crs, always_xy=True)
    xx, yy = transformer.transform(lon, lat)
    row, col = rds.index(xx, yy)


size = 5
image_nparr_test = image.copy()
image_nparr_test[row-size:row+size, col-size:col+size] = image.max()
imshow(image_nparr_test)


# =============================================================================
# ---- plotting all lat long points on the image
# =============================================================================

def pixel_value_of_geo_lat_long(lon, lat, filename):
    with rasterio.open(filename) as rds:
        # convert coordinate to raster projection
        transformer = Transformer.from_crs("EPSG:4326", rds.crs, always_xy=True)
        xx, yy = transformer.transform(lon, lat)
        row, col = rds.index(xx, yy)
    return row, col


csv_filename = r"\\172.20.125.61\mtech\Major Jarmal Singh\Data_Demmin_2015_10_01\Locations_Demmin_west.csv"
df_lat_longz = pd.read_csv(csv_filename, encoding='unicode_escape')

image_nparr_test = image.copy()

for index, rowz in df_lat_longz.iterrows():
    row, col = pixel_value_of_geo_lat_long(rowz["Long."], rowz["Lat."], filename)
    image_nparr_test[row-size:row+size, col-size:col+size] = image_nparr_test.max()
    print(rowz["Lat."],",",rowz["Long."],",",image_nparr_test[row, col][0])

imshow(image_nparr_test)



