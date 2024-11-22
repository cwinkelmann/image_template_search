from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List
import yaml

# Define how to represent Path objects in YAML
def path_representer(dumper, data):
    return dumper.represent_scalar("!Path", str(data))

# Define how to construct Path objects from YAML
def path_constructor(loader, node):
    value = loader.construct_scalar(node)
    return Path(value)

# Create a custom YAML Dumper and Loader
class PathDumper(yaml.Dumper):
    pass

class PathLoader(yaml.Loader):
    pass

def persist_file(file_path: Path, config: dataclass):
    # Register the representer and constructor with the custom Dumper and Loader
    PathDumper.add_representer(Path, path_representer)
    PathLoader.add_constructor("!Path", path_constructor)

    config_dict = asdict(config)
    processed_data = serialize_paths(config_dict)
    with open(file_path, "w") as file:
        yaml.dump(processed_data, file, default_flow_style=False, Dumper=PathDumper)


def load_yaml_config(yaml_file_path: Path, cls: dataclass):
    PathLoader.add_constructor("!Path", path_constructor)
    PathDumper.add_representer(Path, path_representer)

    with open(yaml_file_path, 'r') as file:
        data= yaml.load(file, Loader=PathLoader)

    # Convert string paths back to Path objects
    data = {k: Path(v) if k.endswith('_path') or k == 'base_path' else v for k, v in data.items()}

    # Create WorkflowConfiguration instance
    return cls(**data)

# Convert data to a dictionary and ensure all Path objects are converted to strings
def serialize_paths(data):
    if isinstance(data, dict):
        return {k: serialize_paths(v) for k, v in data.items()}
    elif isinstance(data, Path):
        return str(data)
    elif isinstance(data, list):
        return [serialize_paths(item) for item in data]
    else:
        return data

@dataclass
class WorkflowConfiguration:
    """
    Configuration for the workflow of matching a drone image to an orthomosaic, then the labels
    """
    base_path: Path
    image_url: str
    drone_image_path: Path
    annotations_file_path: Path
    orthomosaic_path: Path
    interm_path: Path
    tile_base_path: Path
    cache_path: Path
    output_path: Path
    buffer_distance: int


class WorkflowReportConfiguration(WorkflowConfiguration):
    combined_annotations_file_path: Path
    buffer_geojson_path: Path
    orthomosaic_proj_path: Path
    orthomosaic_crop_path: Path
    projected_image_2_path: Path
    projected_annotation_path: Path
    combined_annotations_file_path: Path
    dataset_name: str
