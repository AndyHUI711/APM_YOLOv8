import argparse
import cv2
import os

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import platform
import numpy as np
from pathlib import Path
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'ultralytics') not in sys.path:
    sys.path.append(str(ROOT / 'ultralytics'))  # add yolov8 ROOT to PATH
if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strongsort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.yolo.data.dataloaders.stream_loaders import LoadPilAndNumpy
from ultralytics.yolo.utils import LOGGER, SETTINGS, callbacks, colorstr, ops
from ultralytics.yolo.utils.checks import check_file, check_imgsz, check_imshow, print_args, check_requirements
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.ops import Profile, non_max_suppression, scale_boxes, process_mask, process_mask_native
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

from trackers.multi_tracker_zoo import create_tracker


@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov8m.pt',  # model.pt path(s),
        reid_weights=WEIGHTS / 'osnet_x1_0_imagenet.pth',  # model.pt path,
        tracking_method='bytetrack',
        tracking_config=None,
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=100,  # maximum detections per image
        device=0,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_trajectories=False,  # save trajectories for each track
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=0,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs' / 'track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        retina_masks=False,
        line1=125,
        line2=250,
        region_type='both',
        do_entrance_counting=True,
        exp_name='yolov8_infer',
        is_seg=False,
        model=None,
        stride=None,
        names=None,
        pt=None,
        tracker_list=[],
        entrance=None,
        id_set=None,
        interval_id_set=None,
        in_id_list=None,
        out_id_list=None,
        prev_center=None,
        records=None,
        seen=0

):
    source = source

    # Dataloader
    bs = 1

    dataset = LoadPilAndNumpy(
        im0=source,
        imgsz=imgsz,
        stride=stride,
        auto=pt,
        transforms=getattr(model.model, 'transforms', None))

    vid_path, vid_writer, txt_path = [None] * bs, [None] * bs, [None] * bs
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup

    # Create as many strong sort instances as there are video source
    # tracker_list = []
    for i in range(bs):
        tracker = create_tracker(tracking_method, tracking_config, reid_weights, device, half)
        tracker_list.append(tracker, )
        if hasattr(tracker_list[i], 'model'):
            if hasattr(tracker_list[i].model, 'warmup'):
                tracker_list[i].model.warmup()

    outputs = [None] * bs

    # Run tracking
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    windows, dt = [], (Profile(), Profile(), Profile(), Profile())
    curr_frames, prev_frames = [None] * bs, [None] * bs

    for frame_num, batch in enumerate(dataset):
        path, im, im0s, vid_cap, s = batch
        visualize = False
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            # yolov8 predict
            preds = model(im, augment=augment, visualize=visualize)

        # Apply NMS
        with dt[2]:
            if is_seg:
                masks = []
                p = non_max_suppression(preds[0], conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                # nc number of classes
                proto = preds[1][-1]
            else:
                p = non_max_suppression(preds, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process detections
        for i, det in enumerate(p):  # detections per image
            seen += 1

            # p is a fake path
            p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
            # p = Path(p)  # to Path

            curr_frames[i] = im0

            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0[0], line_width=line_thickness, example=str(names))

            if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
                if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                    tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0[0].shape).round()  # rescale boxes to im0 size

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # pass detections to strongsort
                with dt[3]:
                    outputs[i] = tracker_list[i].update(det.cpu(), im0)

                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    # print(outputs[i]) # ONLY WHEN PERSON DETECTED

                    for j, (output) in enumerate(outputs[i]):
                        bbox = output[0:4]
                        id = output[4]
                        cls = output[5]
                        conf = output[6]

                        # to MOT format
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2] - output[0]
                        bbox_h = output[3] - output[1]
                        tlwh = [bbox_left, bbox_top, bbox_w, bbox_h]

                    # add annotator -> info to image
                    if True:  # Add bbox/seg to image
                        c = int(cls)  # integer class
                        id = int(id)  # integer id
                        label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                                              (
                                                                  f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                        color = colors(c, True)
                        center_x = bbox_left + bbox_w / 2.
                        center_y = bbox_top + bbox_h / 2.
                        annotator.box_label(bbox, label, color=color)
                        annotator.circle((int(center_x), int(center_y)), radius=4, color=color)

                        # MOT results
                        tlwh_mot = [tlwh]
                        conf_mot = [conf]
                        id_mot = [id]
                        mot_result = [seen + 1, tlwh_mot, conf_mot, id_mot]
                        # entrance counting
                        statistic = human_flow_counting(True,
                                                        mot_result,
                                                        entrance,
                                                        region_type,
                                                        id_set,
                                                        interval_id_set,
                                                        in_id_list,
                                                        out_id_list,
                                                        prev_center,
                                                        records,
                                                        30,
                                                        2
                                                        )
                        records = statistic['records']
                        annotator.record(records)
            else:
                pass
                # tracker_list[i].tracker.pred_n_update_all_tracks()

            # add lines to image
            entrance_line = tuple(map(int, entrance))
            annotator.box_label(entrance_line[0:4], "DOOR1", color=(0, 0, 255))
            annotator.box_label(entrance_line[4:8], "DOOR2", color=(255, 0, 0))

            # Stream results
            im0 = annotator.result()

            prev_frames[i] = curr_frames[i]

    # Print detection results:
    # Print results
    t = tuple(x.t / 1 * 1E3 for x in dt)  # speeds per image
    # Print total time (preprocessing + inference + NMS + tracking)
    LOGGER.info(
        f"{s}{'' if len(det) else '(no detections), '}{sum([dt.dt for dt in dt if hasattr(dt, 'dt')]) * 1E3:.1f}ms")
    LOGGER.info(
        f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {tracking_method} update per image at shape {(1, 3, *imgsz)}' % t)

    return im0


def human_flow_counting(do_entrance_counting,
                        result,
                        entrance,
                        region_type,
                        id_set,
                        interval_id_set,
                        in_id_list,
                        out_id_list,
                        prev_center,
                        records,
                        video_fps,
                        secs_interval
                        ):
    # Count in/out number:
    if do_entrance_counting:
        assert region_type in [
            'both', 'right', 'left', 'close'
        ], "region_type should be 'horizontal' or 'vertical' or 'custom_1' or 'custom_2' when do entrance counting."

        # test
        # print(f"Test:{region_type} && {entrance}")

        if region_type == 'left' or region_type == 'right':
            entrance_x, entrance_y = entrance[0], entrance[1]
        else:
            entrance_x1, entrance_y1 = entrance[0], entrance[1]
            entrance_x2, entrance_y2 = entrance[4], entrance[5]

        # print(entrance_x, entrance_y)
        frame_id, tlwhs, tscores, track_ids = result
        for tlwh, score, track_id in zip(tlwhs, tscores, track_ids):
            if track_id < 0: continue
            x1, y1, w, h = tlwh
            center_x = x1 + w / 2.
            center_y = y1 + h / 2.
            if track_id in prev_center:
                if region_type == 'left':
                    # left line
                    if prev_center[track_id][1] >= entrance_y and \
                            center_y < entrance_y:
                        in_id_list.append(track_id)
                    if prev_center[track_id][1] <= entrance_y and \
                            center_y > entrance_y:
                        out_id_list.append(track_id)
                elif region_type == 'right':
                    # right line
                    if prev_center[track_id][1] <= entrance_y and \
                            center_y > entrance_y:
                        in_id_list.append(track_id)
                    if prev_center[track_id][1] >= entrance_y and \
                            center_y < entrance_y:
                        out_id_list.append(track_id)
                elif region_type == 'both':
                    # horizontal customized center lines
                    # print(entrance_x1, entrance_y1,entrance_x2, entrance_y2)
                    if prev_center[track_id][1] <= entrance_y1 and \
                            center_y > entrance_y1:
                        in_id_list.append(track_id)
                    if prev_center[track_id][1] >= entrance_y1 and \
                            center_y < entrance_y1:
                        out_id_list.append(track_id)
                    if prev_center[track_id][1] <= entrance_y2 and \
                            center_y > entrance_y2:
                        out_id_list.append(track_id)
                    if prev_center[track_id][1] >= entrance_y2 and \
                            center_y < entrance_y2:
                        in_id_list.append(track_id)
                else:
                    continue
                prev_center[track_id][0] = center_x
                prev_center[track_id][1] = center_y
            else:
                prev_center[track_id] = [center_x, center_y]

        # Count totol number, number at a manual-setting interval
        frame_id, tlwhs, tscores, track_ids = result
        for tlwh, score, track_id in zip(tlwhs, tscores, track_ids):
            if track_id < 0: continue
            id_set.add(track_id)
            interval_id_set.add(track_id)

        # Reset counting at the interval beginning
        if frame_id % video_fps == 0 and frame_id / video_fps % secs_interval == 0:
            curr_interval_count = len(interval_id_set)
            interval_id_set.clear()
        info = "Frame id: {}, Total count: {}".format(frame_id, len(id_set))
        if do_entrance_counting:
            info += ", In count: {}, Out count: {}".format(
                len(in_id_list), len(out_id_list))
        if frame_id % video_fps == 0 and frame_id / video_fps % secs_interval == 0:
            info += ", Count during {} secs: {}".format(secs_interval,
                                                        curr_interval_count)
            interval_id_set.clear()
        # print(info)
        info += "\n"
        records.append(info)

        return {
            "id_set": id_set,
            "interval_id_set": interval_id_set,
            "in_id_list": in_id_list,
            "out_id_list": out_id_list,
            "prev_center": prev_center,
            "records": records,
        }


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=Path, default=WEIGHTS / 'YOLOv8_best.engine',
                        help='model.pt path(s)')
    parser.add_argument('--reid-weights', type=Path, default=WEIGHTS / 'osnet_x1_0_imagenet.pth')
    parser.add_argument('--tracking-method', type=str, default='bytetrack',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--tracking-config', type=Path, default=None)
    parser.add_argument('--source', type=str, default='0',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-trajectories', action='store_true', help='save trajectories for each track')
    parser.add_argument('--save-vid', default=True, action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--retina-masks', action='store_true', help='whether to plot masks in native resolution')

    # entrance count
    parser.add_argument(
        "--line1",
        type=str,
        default=125,
        help="'horizontal' line for entrance counting or break in counting"
    )
    parser.add_argument(
        "--line2",
        type=str,
        default=250,
        help="'horizontal' line for entrance counting or break in counting"
    )
    parser.add_argument(
        "--region-type",
        type=str,
        default='both',
        help="Area type for entrance counting or break in counting"
    )
    parser.add_argument(
        "--do_entrance_counting",
        action='store_true',
        default=True,
        help="Whether counting the numbers of identifiers entering "
    )

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    opt.tracking_config = ROOT / 'trackers' / opt.tracking_method / 'configs' / (opt.tracking_method + '.yaml')
    print_args(vars(opt))
    return opt


def main(opt):
    # check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
