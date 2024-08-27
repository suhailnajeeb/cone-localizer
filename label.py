import os
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM

DATASET_DIR_PATH = '/path/to/dataset/<DATASET_NAME>'

HOME = os.getcwd()

IMAGE_DIR_PATH = '/path/to/all_images'

ontology = CaptionOntology({
    "Traffic Cone": "Cone"
})

base_model = GroundedSAM(ontology = ontology)

dataset = base_model.label(
    input_folder = IMAGE_DIR_PATH,
    extension = ".jpg",
    output_folder = DATASET_DIR_PATH,
)

print("Dataset annotation complete.")