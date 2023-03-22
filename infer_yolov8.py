import os

from ultralytics import YOLO
from cfg_utils import argsparser,merge_cfg,print_arguments
from PIL import Image
import cv2

model = YOLO("yolov8n.pt")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
results_1 = model.predict(source="0")
results_2 = model.predict(source="1")


class YOLO(object):
    """
    YOLO

    Args:
        args (argparse.Namespace): arguments in YOLO, which contains environment and runtime settings
        cfg (dict): config of models in YOLO
    """

    def __init__(self, args, cfg):
        self.multi_camera = True
        self.output_dir = args.output_dir
        self.input = args.camera_id

        if self.multi_camera:
            self.predictor = []
            for name in self.input:
                predictor_item = PipePredictor(
                    args, cfg, is_video=True, multi_camera=True)
                predictor_item.set_file_name(name)
                self.predictor.append(predictor_item)

        else:
            self.predictor = PipePredictor(args, cfg, self.is_video)
            if self.is_video:
                self.predictor.set_file_name(self.input)

    """
    read parameters settings 
    return input
    """

    def _parse_input(self, camera_id):
        # parse input as is_video and multi_camera
        if camera_id != -1:
            print("Use camera id: {}".format(camera_id))
            print(f"Camera 1: No {camera_id[0]}")
            print(f"Camera 2: No {camera_id[1]}")

            if len(camera_id) == 1:
                self.multi_camera = False
            else:
                self.multi_camera = True
            input = camera_id
            self.is_video = True
        else:
            raise ValueError(
                "Illegal Input, please set one of ['camera_id']"
            )

        return input

    def run_multithreads(self):
        import threading
        if self.multi_camera:
            multi_res = []
            threads = []
            for idx, (predictor, input) in enumerate(zip(self.predictor, self.input)):
                thread = threading.Thread(
                    name=str(idx).zfill(3),
                    target=predictor.run,
                    args=(input, idx))
                threads.append(thread)

            for thread in threads:
                thread.start()

            for predictor, thread in zip(self.predictor, threads):
                thread.join()
                collector_data = predictor.get_result()
                multi_res.append(collector_data)

        else:
            self.predictor.run(self.input)

    def run(self):
        if self.multi_camera:
            multi_res = []
            for predictor, input in zip(self.predictor, self.input):
                predictor.run(input)
                collector_data = predictor.get_result()
                multi_res.append(collector_data)
        else:
            self.predictor.run(self.input)


class PipePredictor(object):
    """
    The pipeline for video input:

        1. Tracking
        2. Tracking -> Attribute
        3. Tracking -> KeyPoint -> SkeletonAction Recognition
        4. VideoAction Recognition

    Args:
        args (argparse.Namespace): arguments in pipeline, which contains environment and runtime settings
        cfg (dict): config of models in pipeline
        is_video (bool): whether the input is video, default as False
        multi_camera (bool): whether to use multi camera in pipeline,
            default as False
    """

    def __init__(self, args, cfg, is_video=True, multi_camera=False):

        self.is_video = is_video
        self.multi_camera = multi_camera
        self.cfg = cfg

        self.output_dir = args.output_dir
        self.draw_center_traj = args.draw_center_traj
        self.do_entrance_counting = args.do_entrance_counting
        self.do_break_in_counting = args.do_break_in_counting
        self.region_polygon = args.region_polygon
        self.play_local = args.play_local

        self.warmup_frame = self.cfg['warmup_frame']
        # yolo v8
        self.model = YOLO("yolov8n.pt")


    def set_file_name(self, path):
        if path is not None:
            try:
                self.file_name = os.path.split(path)[-1]
                if "." in self.file_name:
                    self.file_name = self.file_name.split(".")[-2]
            except TypeError:
                self.file_name = None
        else:
            # use camera id
            self.file_name = None

    def get_result(self):
        return self.collector.get_res()

    def run(self, input, thread_idx=0):
        if self.is_video:
            self.predict_video(input, thread_idx=thread_idx)
        else:
            raise ValueError(
                "Illegal Input, please set camera input"
            )

    def predict_video(self, video_file, thread_idx=0):
        print(f"input camera {video_file}, thread_inx {thread_idx} ")
        capture = cv2.VideoCapture(video_file)

        # Get Video info : resolution, fps, frame count
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        print("video fps: %d, frame_count: %d" % (fps, frame_count))
        if self.play_local:
            print("play local using opencv")
        else:
            print(f"Result save to {self.output_dir}")

        frame_id = 0

        entrance, records, center_traj = None, None, None
        if self.draw_center_traj:
            center_traj = [{}]
        id_set = set()
        interval_id_set = set()
        in_id_list = list()
        out_id_list = list()
        prev_center = dict()
        records = list()

        # customize door position # entrance counting
        if self.do_entrance_counting or self.do_break_in_counting or self.illegal_parking_time != -1:
            if self.region_type == 'both':
                entrance = [0, self.region_line1, width, self.region_line1, 0, self.region_line2, width,
                            self.region_line2]

            elif self.region_type == 'right':
                entrance = [0, self.region_line2, width, self.region_line2]

            elif self.region_type == 'left':
                entrance = [0, self.region_line1, width, self.region_line1]

            elif self.region_type == 'close':
                entrance = [0, 0, 0, 0, 0, 0, 0, 0]

            else:
                raise ValueError("region_type:{} unsupported.".format(
                    self.region_type))

        video_fps = fps

        video_action_imgs = []

        object_in_region_info = {
        }  # store info for vehicle parking in region
        illegal_parking_dict = None
        """
        while function
        reading each frame and predict
        """
        # Reading frames
        while True:
            if frame_id % 10 == 0:
                print('Thread: {}; frame id: {}'.format(thread_idx, frame_id))

            ret, frame = capture.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame_id > self.warmup_frame:
                self.pipe_timer.total_time.start()

            # main infer
            if frame_id > self.warmup_frame:
                self.pipe_timer.module_time['mot'].start()

            mot_skip_frame_num = self.mot_predictor.skip_frame_num
            reuse_det_result = False
            if mot_skip_frame_num > 1 and frame_id > 0 and frame_id % mot_skip_frame_num > 0:
                reuse_det_result = True
            res = self.mot_predictor.predict_image(
                [copy.deepcopy(frame_rgb)],
                visual=False,
                reuse_det_result=reuse_det_result)

            # mot output format: id, class, score, xmin, ymin, xmax, ymax
            mot_res = parse_mot_res(res)
            if frame_id > self.warmup_frame:
                self.pipe_timer.module_time['mot'].end()
                self.pipe_timer.track_num += len(mot_res['boxes'])

            if frame_id % 10 == 0:
                print("Thread: {}; trackid number: {}".format(
                    thread_idx, len(mot_res['boxes'])))

            # flow_statistic only support single class MOT
            boxes, scores, ids = res[0]  # batch size = 1 in MOT
            mot_result = (frame_id + 1, boxes[0], scores[0],
                          ids[0])  # single class
            statistic = flow_statistic(
                mot_result,
                self.secs_interval,
                self.do_entrance_counting,
                self.do_break_in_counting,
                self.region_type,
                video_fps,
                entrance,
                id_set,
                interval_id_set,
                in_id_list,
                out_id_list,
                prev_center,
                records,
                ids2names=self.mot_predictor.pred_config.labels)
            records = statistic['records']




            self.collector.append(frame_id, self.pipeline_res)

            # warmup function
            if frame_id > self.warmup_frame:
                self.pipe_timer.img_num += 1
                self.pipe_timer.total_time.end()
            frame_id += 1

            # play local (screen) setting & coding
            if self.cfg['visual']:
                _, _, fps = self.pipe_timer.get_total_time()

                im = self.visualize_video(frame, self.pipeline_res,
                                          self.collector, frame_id, fps,
                                          entrance, records, center_traj,
                                          self.illegal_parking_time != -1,
                                          illegal_parking_dict)  # visualize

                # results display -> screen
                if self.play_local:
                    # Read until video is completed
                    print(f"fps {fps}")
                    # print(f"records {records}")
                    while (capture.isOpened()):
                        # Display the resulting frame
                        cv2.imshow(str(thread_idx), im)
                        # Press Q on keyboard to  exit
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            break
                        # Break the loop
                        else:
                            break
                else:
                    print(f"fps {fps}")
                # elif len(self.pushurl) > 0:
                #     pushstream.pipe.stdin.write(im.tobytes())
                # else:
                #     writer.write(im)
                #     if self.file_name is None:  # use camera_id
                #         cv2.imshow('Paddle-Pipeline', im)
                #         if cv2.waitKey(1) & 0xFF == ord('q'):
                #             break

        # When everything done, release the video capture object
        capture.release()
        # Closes all the frames
        cv2.destroyAllWindows()


    def visualize_video(self,
                        image,
                        result,
                        collector,
                        frame_id,
                        fps,
                        entrance=None,
                        records=None,
                        center_traj=None,
                        do_illegal_parking_recognition=False,
                        illegal_parking_dict=None):
        mot_res = copy.deepcopy(result.get('mot'))
        if mot_res is not None:
            ids = mot_res['boxes'][:, 0]
            scores = mot_res['boxes'][:, 2]
            boxes = mot_res['boxes'][:, 3:]
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        else:
            boxes = np.zeros([0, 4])
            ids = np.zeros([0])
            scores = np.zeros([0])

        # single class, still need to be defaultdict type for ploting
        num_classes = 1
        online_tlwhs = defaultdict(list)
        online_scores = defaultdict(list)
        online_ids = defaultdict(list)
        online_tlwhs[0] = boxes
        online_scores[0] = scores
        online_ids[0] = ids
        # print mot_res for testing
        # print(mot_res)
        # {'boxes': array([[9.00000000e+00, 0.00000000e+00, 7.61203766e-01, 2.49559662e+02,
        # 9.08416733e-02, 3.11986725e+02, 2.35077081e+02]])}

        if mot_res is not None:
            # print(entrance)
            image = plot_tracking_dict(
                image,
                num_classes,
                online_tlwhs,
                online_ids,
                online_scores,
                frame_id=frame_id,
                fps=fps,
                ids2names=self.mot_predictor.pred_config.labels,
                do_entrance_counting=self.do_entrance_counting,
                do_break_in_counting=self.do_break_in_counting,
                do_illegal_parking_recognition=do_illegal_parking_recognition,
                illegal_parking_dict=illegal_parking_dict,
                entrance=entrance,
                records=records,
                center_traj=center_traj)
        else:
            entrance = [0, self.region_line1, 640, self.region_line1, 0, self.region_line2, 640, self.region_line2]
            image = plot_tracking(
                image,
                1,
                frame_id=frame_id,
                fps=fps,
                ids2names=self.mot_predictor.pred_config.labels,
                do_entrance_counting=self.do_entrance_counting,
                do_break_in_counting=self.do_break_in_counting,
                do_illegal_parking_recognition=do_illegal_parking_recognition,
                illegal_parking_dict=illegal_parking_dict,
                entrance=entrance,
                records=records,
                center_traj=center_traj
            )

        human_attr_res = result.get('attr')
        if human_attr_res is not None:
            boxes = mot_res['boxes'][:, 1:]
            human_attr_res = human_attr_res['output']
            image = visualize_attr(image, human_attr_res, boxes)
            image = np.array(image)

        if mot_res is not None:
            vehicleplate = False
            plates = []
            for trackid in mot_res['boxes'][:, 0]:
                plate = collector.get_carlp(trackid)
                if plate != None:
                    vehicleplate = True
                    plates.append(plate)
                else:
                    plates.append("")
            if vehicleplate:
                boxes = mot_res['boxes'][:, 1:]
                image = visualize_vehicleplate(image, plates, boxes)
                image = np.array(image)

        kpt_res = result.get('kpt')
        if kpt_res is not None:
            image = visualize_pose(
                image,
                kpt_res,
                visual_thresh=self.cfg['kpt_thresh'],
                returnimg=True)

        video_action_res = result.get('video_action')
        if video_action_res is not None:
            video_action_score = None
            if video_action_res and video_action_res["class"] == 1:
                video_action_score = video_action_res["score"]
            mot_boxes = None
            if mot_res:
                mot_boxes = mot_res['boxes']
            image = visualize_action(
                image,
                mot_boxes,
                action_visual_collector=None,
                action_text="SkeletonAction",
                video_action_score=video_action_score,
                video_action_text="Fight")

        visual_helper_for_display = []
        action_to_display = []

        skeleton_action_res = result.get('skeleton_action')
        if skeleton_action_res is not None:
            visual_helper_for_display.append(self.skeleton_action_visual_helper)
            action_to_display.append("Falling")

        det_action_res = result.get('det_action')
        if det_action_res is not None:
            visual_helper_for_display.append(self.det_action_visual_helper)
            action_to_display.append("Smoking")

        cls_action_res = result.get('cls_action')
        if cls_action_res is not None:
            visual_helper_for_display.append(self.cls_action_visual_helper)
            action_to_display.append("Calling")

        if len(visual_helper_for_display) > 0:
            image = visualize_action(image, mot_res['boxes'],
                                     visual_helper_for_display,
                                     action_to_display)

        return image



def main():
    cfg = merge_cfg(FLAGS)  # use command params to update config
    print_arguments(cfg)

    yolorun = YOLO(FLAGS, cfg)
    # pipeline.run()
    yolorun.run_multithreads()


if __name__ == '__main__':
    # parse params from command
    parser = argsparser()
    FLAGS = parser.parse_args()
    FLAGS.device = FLAGS.device.upper()
    assert FLAGS.device in ['CPU', 'GPU', 'XPU'
                            ], "device should be CPU, GPU or XPU"
    main()
