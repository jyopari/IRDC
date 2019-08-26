# IRDC
Code for Semi-Automated Computa
tional Approach for Infrared Dark Cloud Localization: A Catalog of Infrared Dark Clouds, by Jyo Pari and Joseph L. Hora 
# How to use
First create 3 empty folders, which are "/input", "/images", "/results"

Add your FITS file in the input folder, and then run this command: python3 main.py pathtofits C0 C1 T0 T1 T2 0 or 1 depending on whether you want to remove the false positives near the top. For example, if you are running GLM_00000+0000_I1.fits using the parameters in our paper, the command would be: python3 main.py input/GLM_00000+0000_mosaic_I1.fits 0.75 0.00 6 15 40 1
