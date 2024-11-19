# image_template_search
find template images in larger image sets. This allows for the following workflows:

- find a small template image in a larger image



## Installation
```bash
conda create -n image-template-search python=3.11
```
install LightGlue first

```bash
git clone --quiet https://github.com/cvg/LightGlue/
cd LightGlue

# install the package without output
pip install --progress-bar off --quiet -e .

# alternative to the above
pip install -e .
```

Then install the requirements
```bash
pip install -r requirements.txt
```

## Example Workflows


### find a small template image t in a larger image l


### create a database of images
image_database.py

read the metadata of images


### image_template_search/detection_deduplication.py
Iterate through all images and their annotations

### clip a buffer around an image location from an orthomosaic
To do this a bigger Orthomosaic is cut to the area of the drone image.
clip_orthomosaic_by_drone_image_loc.py

### Find image patches in bigger orthomosaics
Use an template image, an orthomosaic and hasty annotation to map all marks to a projected drone image.

opencv_drone_to_orthomosaic.py


This template from: 

There are 3 set of orthomosaics, with for mosaics each from metashape, pix4d, dronedeploy, opendronemap
- ../IguanasFromAbove/Orthomosaics for quality analysis/<Mission_Name>/
