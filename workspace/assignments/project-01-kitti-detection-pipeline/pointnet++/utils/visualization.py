import numpy as np
import open3d as o3d


def get_bounding_boxes(segmented_objects, object_ids, predictions, decoder):
    """
    Draw bounding boxes for surrounding objects according to classification result
        - red for pedestrian
        - blue for cyclist
        - green for vehicle

    Parameters
    ----------
    segmented_objects: open3d.geometry.PointCloud
        Point cloud of segmented objects
    object_ids: numpy.ndarray
        Object IDs as numpy.ndarray
    predictions:
        Object Predictions

    Returns
    ----------

    """
    # parse params:
    points = np.asarray(segmented_objects.points)

    # color cookbook:
    color = {
        # pedestrian as red:
        'pedestrian': np.asarray([0.5, 0.0, 0.0]),
        # cyclist as blue:
        'cyclist': np.asarray([0.0, 0.0, 0.5]),
        # vehicle as green:
        'vehicle': np.asarray([0.0, 0.5, 0.0]),
    }

    bounding_boxes = []
    for class_id in predictions:
        # get color
        class_name = decoder[class_id]

        if (class_name == 'misc'):
            continue

        class_color = color[class_name]
        # show instances:
        for object_id in predictions[class_id]:
            # create point cloud:
            object_pcd = o3d.geometry.PointCloud()
            object_pcd.points = o3d.utility.Vector3dVector(
                points[object_ids == object_id]
            )
            # create bounding box:
            bounding_box = object_pcd.get_axis_aligned_bounding_box()

            # set color according to confidence:
            confidence = predictions[class_id][object_id]
            bounding_box.color = tuple(
                class_color + (1.0 - confidence)*class_color
            )

            # update:
            bounding_boxes.append(bounding_box)
    
    return bounding_boxes