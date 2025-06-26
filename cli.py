import click
from pathlib import Path
from image_template_search.types.workflow_config import WorkflowConfiguration, load_yaml_config
from image_template_search.workflow_iguana_deduplication import workflow_project_single_image_drone_and_annotations


@click.command()
@click.argument('file_path', type=click.Path(exists=True, dir_okay=False, path_type=Path))
def process_image(file_path):
    """
    CLI to process a single drone image with annotations.

    \b
    Arguments:
    FILE_PATH - Path to the YAML configuration file.
    """
    try:
        # Load the workflow configuration from the YAML file
        config = load_yaml_config(yaml_file_path=file_path, cls=WorkflowConfiguration)

        # Run the workflow
        workflow_project_single_image_drone_and_annotations(config)
        click.echo("Workflow executed successfully.")

    except Exception as e:
        click.echo(f"An error occurred: {e}", err=True)

if __name__ == '__main__':
    process_image()