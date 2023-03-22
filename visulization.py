import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def plot_tracking(image,
                       num_classes,
                       frame_id=0,
                       fps=0.,
                       ids2names=[],
                       do_entrance_counting=False,
                       entrance=None,
                       records=None,
                       center_traj=None):

    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    text_scale = max(0.5, image.shape[1] / 3000.)
    text_thickness = 2
    line_thickness = max(1, int(image.shape[1] / 500.))
    records = None
    if num_classes == 1:
        if records is not None:
            start = records[-1].find('Total')
            end = records[-1].find('In')
            cv2.putText(
                im,
                records[-1][start:end], (0, int(40 * text_scale) + 10),
                cv2.FONT_ITALIC,
                text_scale, (0, 0, 255),
                thickness=text_thickness)

    if do_entrance_counting:
        entrance_line = tuple(map(int, entrance))
        cv2.rectangle(
            im,
            entrance_line[0:2],
            entrance_line[2:4],
            color=(0, 255, 255),
            thickness=line_thickness)
        cv2.rectangle(
            im,
            entrance_line[4:6],
            entrance_line[6:8],
            color=(255, 0, 255),
            thickness=line_thickness)



    return im