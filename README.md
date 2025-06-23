# image_template_search
find template images in larger image sets. This allows for the following workflows:

- find a small template image in a larger image



## Installation
```bash
conda create -n image_template_search python=3.11
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

# run the tests
pytest tests
```

Finally, install the image_template_search package
```

## Example Workflows


### find a small template image t in a larger image l
TODO: describe the simple steps a smaller image in a larger image
There are extension like, projecting annotations too.

### Using a stack of images, the following steps are necessary:
- find nearby imgage
- match a crop from the primary image to the nearby images
- (optional) project annotations from the primary image to the nearby images
- (optional) use annotations on all images to estimate the total amount of objects

### image comparison workflow
To evaluate if an orthomosaic matches the quality of a drone image, the following steps are necessary:
1. config_generation_batch_template_matching.py
2. batched_template_matching.py
3. batched_human_in_the_loop.py
4. TODO: batched_calculate_metrics.py

### create a database of images
image_database.py

### Register two Orthomosaics
See register_two_orthomosaics.py
and workflow_register_two_orthomosaics.py


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
