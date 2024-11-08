# image_template_search
find template images in larger image sets. This allows for the following workflows:

- find a small template image in a larger image



## Installation
```bash
conda create -n image-template-search python=3.11
```

TODO: install LightGlue first

```bash
git clone --quiet https://github.com/cvg/LightGlue/
cd LightGlue
pip install --progress-bar off --quiet -e .
pip install -e .
```


## Workflow
find a small template image t in a larger image l

t ... 1280*1280
l ... 5000 * 4000


t = l ... 5000 * 4000

t ... 5000 * 4000
l ... 50.000 * 50.000