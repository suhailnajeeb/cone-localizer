import os
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM

# Path to the dataset directory
DATASET_DIR_PATH = '/path/to/dataset/<DATASET_NAME>'

# Path to the directory containing all images
IMAGE_DIR_PATH = '/path/to/all_images'

def annotate_dataset(image_dir, dataset_dir, image_extension=".jpg"):
    """
    Annotates a dataset of images using the GroundedSAM model based on a defined ontology.

    This script requires access to a GPU and significant memory resources, depending
    on the number of samples to be annotated. Ensure that the machine running this
    script meets these hardware requirements to avoid performance bottlenecks or crashes.

    Args:
    image_dir (str): Directory containing the images to be annotated.
    dataset_dir (str): Directory where the annotated dataset will be saved.
    image_extension (str): File extension of the images to be annotated. Default is '.jpg'.

    Returns:
    None
    """

    # Define the ontology for labeling the dataset
    ontology = CaptionOntology({
        "Traffic Cone": "Cone"
    })

    # Initialize the GroundedSAM model with the defined ontology
    base_model = GroundedSAM(ontology=ontology)

    # Annotate the images and save to the specified output folder
    dataset = base_model.label(
        input_folder=image_dir,
        extension=image_extension,
        output_folder=dataset_dir,
    )

    print("Dataset annotation complete.")

# Execute the annotation function
annotate_dataset(IMAGE_DIR_PATH, DATASET_DIR_PATH)
