# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 14:15:36 2025

@author: USER
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from osgeo import gdal
from osgeo import osr

wd_path = r"\\172.20.125.61\mtech\Major Jarmal Singh\Data_Chilika_Prisma\Data\PRS_L2D_STD_20191228050235_20191228050239_0001 worked\Output_Tiff_format"


# HSI Image paths
VNIR_image_path = wd_path + "\PRS_L2D_STD_20191228050235_20191228050239_0001_HCO_VNIR.tif"
SWIR_image_path = wd_path + "\PRS_L2D_STD_20191228050235_20191228050239_0001_HCO_SWIR.tif"

# VNIR image
Image_VNIR = rasterio.open(VNIR_image_path)
Crs = Image_VNIR.crs
trans = Image_VNIR.transform
cols = Image_VNIR.width
rows = Image_VNIR.height
bands= Image_VNIR.count
print("rows:", rows, ", cols:", cols, ', bands:', bands)
Image_VNIR = Image_VNIR.read()
Image_VNIR = np.moveaxis(Image_VNIR, 0, -1) # moving number of bands at last


# Plotting Trure color RGB image
plt.imshow(Image_VNIR[:, :, [34, 23, 11]], vmin=-1, vmax=1)
plt.show()


# SWIR image
Image_SWIR = rasterio.open(SWIR_image_path)
Image_SWIR = Image_SWIR.read()
Image_SWIR = np.moveaxis(Image_SWIR, 0, -1) # moving number of bands at last


# Image concatenation
Image_data = np.concatenate((Image_VNIR, Image_SWIR), axis=2)


# NDVI calculation function
def ndvi(img):
    r=img[:, :, 34]
    nir=img[:, :, 121]
    dinom = (nir+r)
    numer = (nir-r)
    ndvi = np.where(dinom==0.0, 0.0, ((numer/dinom)*1.0))
    return ndvi

Image_ndvi = ndvi(Image_data)

plt.imshow(Image_ndvi, cmap="jet", vmin=-1, vmax=1)
plt.colorbar()


# Normalization
Normalized_ndvi = (((Image_ndvi - Image_ndvi.min()) / (Image_ndvi.max() - Image_ndvi.min())) * 255.9).astype(np.uint8) # Normalization
Normalized_ndvi = Image.fromarray(Normalized_ndvi)

plt.imshow(Normalized_ndvi, cmap="jet", vmin=-1, vmax=1)
plt.colorbar()





























