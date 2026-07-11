# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 16:08:30 2022

@author: IISTDBT
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.cm as cm
from osgeo import gdal
from osgeo import osr


wd_path = r"E:\0.Manoj\7.Project SOC Post Validation facto\EnmapData\Adilabad\2022Jul16-1"


# Source image
path_source_image = wd_path + "\\ENMAP01-____L2A-DT0000001781_20220716T055407Z_002_V010502_20251006T222733Z-RGB_IMAGE.tif"
Xs = rasterio.open(path_source_image)
Xs = Xs.read()


# Reading File Information
dataset = rasterio.open(path_source_image)
print (dataset)
Crs = dataset.crs
trans = dataset.transform
cols = dataset.width
rows = dataset.height
bands= dataset.count
print("rows:", rows, ", cols:", cols, ', bands:', bands)


Xs = np.moveaxis(Xs, 0, -1) # moving number of bands at last



# Plotting Gray Scale image
plt.imshow(Xs[:, :, [0]])
plt.show()


# Plotting True color RGB image
plt.imshow(Xs[:, :, [0, 1, 2]])
plt.show()


# NDVI calculation function
# In AVIRIS_NG HyperSpectral Image Red is 52th band and NIR is 118th band
# In PRISMA HyperSpectral Image BGR_Nir bands are is 10th, 18th, 35th, and 122th (approx corrosponding wavelengths are: 475, 535, 684, and 1554 nm respectively)

def ndvi(img):
    r=img[:, :, 52] # For Demmin red band index is different
    nir=img[:, :, 118]
    dinom = (nir+r)
    numer = (nir-r)
    ndvi = np.where(dinom==0.0, 0.0, ((numer/dinom)*1.0))
    plt.imshow(ndvi,cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar()
    return ndvi

Xt_ndvi = ndvi(Xs)

plt.imshow(Xt_ndvi, cmap="jet", vmin=-1, vmax=1)
plt.colorbar()

# Save NDVI Image
path = r"C:\Users\USER\Desktop\Prisma\PRS_L2D_STD_20191228050235_20191228050239_0001\NDVI_n_Mask_Image"

I_ndvi = (((Xt_ndvi - Xt_ndvi.min()) / (Xt_ndvi.max() - Xt_ndvi.min())) * 255.9).astype(np.uint8)
im = Image.fromarray(I_ndvi)
im.save(path+"\\ndvi.png")


# Creating the mask for soil
def create_soil_mask(ndvi_image):
    soil_mask = np.empty([Xt_ndvi.shape[0]*Xt_ndvi.shape[1]])
    for i in range(0, len(ndvi_image)):
        if ndvi_image[i] > 0 and ndvi_image[i] <= 0.3: 
            soil_mask[i] = 1
        else:
            soil_mask[i] = 0
    return soil_mask.reshape(Xt_ndvi.shape[0], Xt_ndvi.shape[1])
            
soil_mask = create_soil_mask(Xt_ndvi.reshape(Xt_ndvi.shape[0]*Xt_ndvi.shape[1]))

plt.imshow(soil_mask, cmap="gray", vmin=-1, vmax=1)
plt.colorbar()

np.unique(soil_mask)

# Save Binary Mask Image
# np.save(path+"\\soil_mask", soil_mask)
plt.imsave(path+"\\soil_mask.png", soil_mask, cmap=cm.gray)


# Overlapping mask one by one on each channel
masked_val = []
for i in range(bands):
    band_array = Xs[:,:,i]
    masked_val.append(np.multiply(band_array, soil_mask))

masked_value_array = np.asanyarray(masked_val)


# rotate channels for just view
masked_value_array = np.moveaxis(masked_value_array,0,-1) # rotating the last
plt.imshow(masked_value_array[:,:,75])

# rerotate the bands for further processing
masked_value_array = np.moveaxis(masked_value_array,0,-1) # rotating the bands
masked_value_array = np.moveaxis(masked_value_array,0,-1) # rotating the bands


# writing the output file
Output_filename = r"C:\Users\USER\Desktop\Prisma\PRS_L2D_STD_20191228050235_20191228050239_0001\Soil_Segmented_image\PRS_L2D_STD_20191228050235_20191228050239_0001.tiff"

dataset = gdal.Open(path_source_image, gdal.GA_ReadOnly)
gdal_datatype = gdal.GDT_Float32
np_datatype = np.float32

driver = gdal.GetDriverByName( "GTiff" )
originX, pixelWidth, b, originY, d, pixelHeight = dataset.GetGeoTransform()
output_file = driver.Create(Output_filename, cols, rows, bands, gdal_datatype)
output_file.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
for i, image in enumerate(masked_value_array, 1):
        output_file.GetRasterBand(i).WriteArray( image )
prj = dataset.GetProjection()
outRasterSRS = osr.SpatialReference(wkt=prj)
output_file.SetProjection(outRasterSRS.ExportToWkt())
output_file.FlushCache()
output_file = None



















