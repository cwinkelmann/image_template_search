"""
mosaics from DroneDeploy are in jpg format. We need to convert them to COG format.
"""
from pathlib import Path

from image_template_search.geospatial_transformations import convert_to_cog, batch_convert_to_cog

input_files = Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Drone Deploy orthomosaics").glob("Snt_STJB*.tif")
# input_files = Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Drone Deploy orthomosaics").glob("Esp_EGI01_13012021*")
input_files = list(input_files)

print(f"Found {len(input_files)} files")
# output_dir = Path("/Users/christian/PycharmProjects/hnee/image_template_search/data/cog/")
output_dir = Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Drone Deploy orthomosaics/cog/all")
output_dir.mkdir(exist_ok=True, parents=True)
# input_files = [Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Drone Deploy orthomosaics/Scruz_SCPLF01_15012023.tif"),
#                Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Drone Deploy orthomosaics/Scris_SRPB02to05_25012020.tif"),
#                Path("/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Drone Deploy orthomosaics/Scris_SRPB06_25012020.tif")]


# for input_file in input_files:
#     output_file = output_dir / f"{Path(input_file).stem}.tif"
#     convert_to_cog(input_file, output_file)

batch_convert_to_cog(input_files, output_dir, max_workers=4)


### Batch command
"""
gdal_translate "/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Drone Deploy orthomosaics/Snt_STJB01to06_12012023-1.tif" "/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Drone Deploy orthomosaics/cog/Snt_STJB01to06_12012023-1_cog_shell_JPG_tiled.tif" -of COG -co COMPRESS=JPEG -co BIGTIFF=YES -co TILED=YES -co BLOCKSIZE=4048
gdal_translate "/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Drone Deploy orthomosaics/Snt_STJB01to05_10012023.tif" "/Volumes/G-DRIVE/Iguanas_From_Above/Manual_Counting/Drone Deploy orthomosaics/cog/Snt_STJB01to05_10012023_cog_shell.tif" -of COG -co COMPRESS=JPEG -co BIGTIFF=YES -co TILED=YES -co BLOCKSIZE=4048
"""