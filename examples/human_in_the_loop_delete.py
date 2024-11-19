"""
take modified annotation from cvat, download them, delete them create a local annotation file


"""
from pathlib import Path
import fiftyone as fo
import shapely
from loguru import logger

from image_template_search.util.HastyAnnotationV2 import hA_from_file, ImageLabel, Keypoint


def retrieve_from_cvat(anno_key, dataset_name = "projection_comparison"):






    return results, view


if __name__ == '__main__':
    ## The image for Andrea:
    base_path = Path("/Users/christian/Library/CloudStorage/GoogleDrive-christian.winkelmann@gmail.com/My Drive/Datasets/IguanasFromAbove/Orthomosaics for quality analysis/")
    images_dir = base_path / "San_STJB01_10012023/template_images/San_STJB01_10012023_DJI_0068"
    drone_image = images_dir / "San_STJB01_10012023_DJI_0068.JPG"
    image_2 =  base_path / "San_STJB01_10012023/San_STJB01_10012023_orthomosaic_DDeploy.tif"

    annotations_file_path = base_path / "San_STJB01_10012023/template_images/methods_paper_labels.json"
    corrected_annotations_file_path = base_path / "San_STJB01_10012023/template_images/methods_paper_labels_corrected.json"
    # load hasty annotations
    hA = hA_from_file(file_path=annotations_file_path)

    hA_corrected = hA_from_file(file_path=annotations_file_path)
    hA_corrected.images = []

    dataset_name = "projection_comparison"
    anno_key = "cvat_basic_recipe"

    cleanup = False

    # create dot annotations
    # results, view = retrieve_from_cvat(anno_key=anno_key, dataset_name = dataset_name)

    # Step 5: Merge annotations back into FiftyOne dataset

    dataset = fo.load_dataset(dataset_name)
    cvat_dataset = dataset.load_annotations(anno_key)

    # Load the view that was annotated in the App
    view = dataset.load_annotation_view(anno_key)

    # session = fo.launch_app(view=view)
    # session.wait()

    # reconstruct an annotation file
    for sample in view:
        filepath = sample.filepath
        hasty_image_id = sample.hasty_image_id
        hasty_filename = sample.filename

        image = [i for i in hA.images if i.image_id == sample.hasty_image_id][0]
        # TODO merge the dataset with original annotations


        updated_labels = []

        for kp in sample.ground_truth_boxes.detections:
            # print(kp)
            x1,y1,w,h = kp.bounding_box
            x2 = x1 + w
            y2 = y1 + h
            x1 *= image.width
            x2 *= image.width
            y1 *= image.height
            y2 *= image.height

            bbox = [int(x1), int(y1), int(x2), int(y2)]
            pt = shapely.box(*bbox).centroid


            if hasattr(kp, "hasty_id"):
                # The object is a known object from before
                logger.info(f"Object {kp.hasty_id} was known before")

                image_label = [l for l in image.labels if l.id == kp.hasty_id][0]



                dist = pt.distance(image_label.centroid)
                if dist > 2:
                    # The object was moved
                    logger.info(f"Object {kp.hasty_id} was moved")

                    image_label.bbox = bbox

                else:
                    # The object was not moved
                    logger.info(f"Object {kp.hasty_id} was not moved")

                updated_labels.append(image_label)


            else:
                # The object is new
                logger.info("New object")
                il = ImageLabel(class_name='iguana',
                           bbox=bbox,
                           polygon=None,
                           mask=None, z_index=0,
                           keypoints=[])
                updated_labels.append(il)


        # for kp in sample.ground_truth.keypoints:
        #     # print(kp)
        #     pt = shapely.Point(kp.points[0][0] * image.width, kp.points[0][1] * image.height)
        #     if hasattr(kp, "hasty_id"):
        #         # The object is a known object from before
        #         # TODO was the annotation moved?
        #         logger.info(f"Object {kp.hasty_id} was known before")
        #
        #         image_label = [l for l in image.labels if l.id == kp.hasty_id][0]
        #         dist = pt.distance(image_label.centroid)
        #         if dist > 2:
        #             # The object was moved
        #             logger.info(f"Object {kp.hasty_id} was moved")
        #
        #
        #         else:
        #             # The object was not moved
        #             logger.info(f"Object {kp.hasty_id} was not moved")
        #             updated_labels.append(image_label)
        #
        #
        #     else:
        #         # The object is new
        #         logger.info("New object")
        #         hkp = Keypoint(x=int(pt.x),
        #                        y=int(pt.y),
        #                  norder=0)
        #         il = ImageLabel(
        #             class_name=kp.label,
        #             keypoints=[hkp],
        #             attributes={"comment": "Created from CVAT"},
        #         )

        image.labels = updated_labels

        hA_corrected.images.append(image)

    with open(corrected_annotations_file_path, 'w') as json_file:
        json_file.write(hA_corrected.model_dump_json())


    if cleanup:
        # Step 6: Cleanup

        ## Delete tasks from CVAT
        results = view.load_annotation_results(anno_key)
        # results.cleanup()

        ##  Delete run record (not the labels) from FiftyOne
        # dataset.delete_annotation_run(anno_key)



            ## HOW to reconstruct the label cvat dataset
            # labels = [
            #     ImageLabel(
            #         category=label.label,
            #         bounding_box=label.bounding_box
            #     ) .detections  # Assuming `ground_truth` field
            # ]
        # session = fo.launch_app(dataset, port=5151)

        # logger.info("Deleting dataset")
        # fo.delete_dataset(dataset_name)