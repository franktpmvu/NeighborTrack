import importlib
import os
from collections import OrderedDict
from lib.test.evaluation.environment import env_settings
import time
import cv2 as cv
import copy
from lib.utils.lmdb_utils import decode_img
from pathlib import Path
import numpy as np
import sys

print('plz change /data/ to your base path')
sys.path.append('/data/')

from NeighborTrack.neighbortrack import neighbortrack
from NeighborTrack.NTutils.utils import  xy_wh_2_rect,pos_sz_2_xywh


def trackerlist(name: str, parameter_name: str, dataset_name: str, run_ids = None, display_name: str = None,
                result_only=False):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, dataset_name, run_id, display_name, result_only) for run_id in run_ids]


class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, dataset_name: str, run_id: int = None, display_name: str = None,
                 result_only=False):
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name

        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
        if result_only:
            self.results_dir = '{}/{}'.format(env.results_path, self.name)

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                              '..', 'tracker', '%s.py' % self.name))
        if os.path.isfile(tracker_module_abspath):
            tracker_module = importlib.import_module('lib.test.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

    def create_tracker(self, params):
        tracker = self.tracker_class(params, self.dataset_name)
        return tracker

    def run_sequence(self, seq, debug=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        # Get init information
        init_info = seq.init_info()

        tracker = self.create_tracker(params)

        output = self._track_sequence(tracker, seq, init_info)
        return output

    def _track_sequence(self, tracker, seq, init_info):
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i

        output = {'target_bbox': [],
                  'time': []}
        if tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = self._read_image(seq.frames[0])

        start_time = time.time()
        out = tracker.initialize(image, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)
        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time}
        if tracker.params.save_all_boxes:
            init_default['all_boxes'] = out['all_boxes']
            init_default['all_scores'] = out['all_scores']

        _store_outputs(out, init_default)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image = self._read_image(frame_path)

            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            if len(seq.ground_truth_rect) > 1:
                info['gt_bbox'] = seq.ground_truth_rect[frame_num]
            out = tracker.track(image, info)
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output

    def run_video(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False, showcv=False, save_img=False):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        # self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"

        output_boxes = []

        cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + tracker.params.tracker_name
        success, frame = cap.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        if showcv:

            cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
            cv.resizeWindow(display_name, 960, 720)
            cv.imshow(display_name, frame)

        def _build_init_info(box):
            return {'init_bbox': box}

        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(frame, _build_init_info(optional_box))
            output_boxes.append(optional_box)
        else:
            if showcv: 
                while True:
                    # cv.waitKey()
                    frame_disp = frame.copy()
                    frame_disp = cv.cvtColor(frame_disp, cv.COLOR_BGR2RGB)

                    cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                               1.5, (0, 0, 0), 1)

                    x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                    init_state = [x, y, w, h]
                    tracker.initialize(frame, _build_init_info(init_state))
                    output_boxes.append(init_state)
                    break
        frame_index=0
        while True:
            
            ret, frame = cap.read()
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            
            if frame is None:
                break

            frame_disp = frame.copy()
            frame_disp = cv.cvtColor(frame_disp, cv.COLOR_BGR2RGB)
            # Draw box
            out = tracker.track(frame)
            state = [int(s) for s in out['target_bbox']]
            output_boxes.append(state)


            font_color = (0, 0, 0)
            if 'all_boxes' in out.keys():
                neighbor_bbox = np.reshape(out['all_boxes'],(-1,4))
            else:
                neighbor_bbox = []
            for n_bbox in neighbor_bbox:
                n_state = [int(s) for s in n_bbox]
                print('---have neighbor info---')
                print(neighbor_bbox)
                print(state)

                cv.rectangle(frame_disp, (n_state[0], n_state[1]), (n_state[2] + n_state[0], n_state[3] + n_state[1]),
                         (255,0 , 0), 5)
                
            cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)


            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)

            # Display the resulting frame
            if showcv:
                cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ret, frame = cap.read()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                           (0, 0, 0), 1)
                if showcv:
                    cv.imshow(display_name, frame_disp)
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)
                
            if save_img:
                video_name = Path(videofilepath).stem
                if not os.path.exists(self.results_dir+'/'+video_name):
                    os.makedirs(self.results_dir+'/'+video_name)

                base_results_path = os.path.join(self.results_dir+'/'+video_name)
                cv.imwrite(base_results_path+'/%08d' %frame_index+'.jpg', frame_disp)
                frame_index = frame_index+1


        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')


    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module('lib.test.parameter.{}'.format(self.name))
        params = param_module.parameters(self.parameter_name)
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")
            
            
class NeighborTracker(Tracker):
    
    def _track_sequence(self, tracker, invtracker, seq, init_info):
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.
        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i
        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i
        output = {'target_bbox': [],
                  'time': []}
        if tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []
        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)
        # Initialize
        image = self._read_image(seq.frames[0])
        
        start_time = time.time()
        out = tracker.initialize(image, init_info)
        #init_info = {'init_bbox': list(gt_bbox_np)}
        region = init_info.get('init_bbox')
        ntracker = neighbortrack(tracker,image,region[:2],region[2:],revtracker=invtracker)
        #ntracker.ls_add_mode = 0
        #ntracker.rev_frames = 27
        #ntracker.test_probe_rev_frames = 27
        #ntracker.del_winner = True

        if out is None:
            out = {}
        prev_output = OrderedDict(out)
        init_default = {'target_bbox': region,
                        'time': time.time() - start_time}
        if tracker.params.save_all_boxes:
            init_default['all_boxes'] = out['all_boxes']
            init_default['all_scores'] = out['all_scores']
        _store_outputs(out, init_default)
        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image = self._read_image(frame_path)
            start_time = time.time()
            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output
            if len(seq.ground_truth_rect) > 1:
                info['gt_bbox'] = seq.ground_truth_rect[frame_num]
                
                
            #out = tracker.track(image, info)
            
            state = ntracker._neighbor_track(image)
            location = xy_wh_2_rect(state['target_pos'], state['target_sz'])
            out = {"target_bbox": location}
            
            
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})
        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)
        return output

    
    def run_sequence(self, seq, debug=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        # Get init information
        init_info = seq.init_info()

        tracker = self.create_tracker(params)
        invtracker = self.create_tracker(params)

        output = self._track_sequence(tracker,invtracker, seq, init_info)
        return output

    def run_video_neighbor(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False, showcv=False, save_img=False,invtracker=None):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """
        if invtracker is None:
            print('no invtracker define')
            dsa

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        # self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)
            invtracker=invtracker.create_tracker(params)

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"

        output_boxes = []

        cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + tracker.params.tracker_name
        success, frame = cap.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        if showcv:

            cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
            cv.resizeWindow(display_name, 960, 720)
            cv.imshow(display_name, frame)

        def _build_init_info(box):
            return {'init_bbox': box}

        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(frame, _build_init_info(optional_box))
            
            ntracker = neighbortrack(tracker,frame,optional_box[:2],optional_box[2:],invtracker=invtracker)

            output_boxes.append(optional_box)
        else:
            if showcv: 
                while True:
                    # cv.waitKey()
                    frame_disp = frame.copy()
                    frame_disp = cv.cvtColor(frame_disp, cv.COLOR_BGR2RGB)

                    cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                               1.5, (0, 0, 0), 1)

                    x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                    init_state = [x, y, w, h]
                    tracker.initialize(frame, _build_init_info(init_state))
                    output_boxes.append(init_state)
                    break
        frame_index=0
        while True:
            
            ret, frame = cap.read()
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            if frame is None:
                break

            frame_disp = frame.copy()
            frame_disp = cv.cvtColor(frame_disp, cv.COLOR_BGR2RGB)
            frame_disp_neighbor = frame_disp.copy()
            # Draw box
            #out = tracker.track(frame)
            state_neighbor = ntracker._neighbor_track(frame)
            print(frame_index)
            if state_neighbor['change_from_neighbor']:
                #print(state_neighbor)
                print(state_neighbor['old_neighbor_pos'])# CAND
                print(state_neighbor['old_neighbor_sz'])# CAND
                print(state_neighbor['rev_target_pos'])# CAND
                print(state_neighbor['rev_target_sz'])# CAND

                #print(state_neighbor['old_neighbor_score'])# CAND
                print(state_neighbor['old_target_pos'])# trackpool_history_target
                print(state_neighbor['old_target_sz'])# trackpool_history_target
                #print(state_neighbor['old_target_score'])# trackpool_history_target
                print(state_neighbor['trackpool_neighbor_pos'])# trackpool_neighbor
                print(state_neighbor['trackpool_neighbor_sz'])# trackpool_neighbor
                #print(state_neighbor['trackpool_neighbor_score'])# trackpool_neighbor
                #print(state_neighbor['old_neighbor_posind'])# CAND
                print(state_neighbor['location_neighbor_pred'])# CAND
                print(state_neighbor['location_pred'])# C_j
                print(state_neighbor['score_pred'])# C_j
                print(len(state_neighbor['oldim']))
                print(state_neighbor['oldim'][0].shape)
                print(state_neighbor['target_pos'])
                print(state_neighbor['target_sz'])
                
                our_xywh = pos_sz_2_xywh(state_neighbor['target_pos'],state_neighbor['target_sz'])
                our_xywh = [int(s) for s in our_xywh]
                original_xywh = state_neighbor['location_pred']
                original_xywh = [int(s) for s in original_xywh]

                #our method
                cv.rectangle(frame_disp_neighbor, (our_xywh[0], our_xywh[1]), (our_xywh[2] + our_xywh[0], our_xywh[3] + our_xywh[1]),
                         (0, 0, 255), 5)
                #original method
                cv.rectangle(frame_disp_neighbor, (original_xywh[0], original_xywh[1]), (original_xywh[2] + original_xywh[0], original_xywh[3] + original_xywh[1]),
                         (255, 0, 0), 5)

                
                cv.imwrite(base_results_path+'/%08d' %frame_index+'neighbors_.jpg', frame_disp_neighbor)

                
                
                old_images = copy.deepcopy(state_neighbor['oldim'])
                for ind,old_image in enumerate(old_images):
                    old_image = cv.cvtColor(old_image, cv.COLOR_BGR2RGB)
                    old_image_ori = old_image.copy()
                    old_image_our = old_image.copy()
                    old_image_hist = old_image.copy()
                    
                    our_xywh = pos_sz_2_xywh(state_neighbor['winner_rev_pos'][ind],state_neighbor['winner_rev_sz'][ind])
                    our_xywh = [int(s) for s in our_xywh]

                    original_xywh = pos_sz_2_xywh(state_neighbor['rev_target_pos'][ind],state_neighbor['rev_target_sz'][ind])
                    original_xywh = [int(s) for s in original_xywh]

                    history_xywh = pos_sz_2_xywh(state_neighbor['old_target_pos'][ind],state_neighbor['old_target_sz'][ind])
                    history_xywh = [int(s) for s in history_xywh]

                    #frame_disp_old_8 = copy.deepcopy(state_neighbor['oldim'][8])
                    cv.rectangle(old_image, (original_xywh[0], original_xywh[1]), (original_xywh[2] + original_xywh[0], original_xywh[3] + original_xywh[1]),
                             (255, 0, 255), 5)
                    cv.rectangle(old_image_ori, (original_xywh[0], original_xywh[1]), (original_xywh[2] + original_xywh[0], original_xywh[3] + original_xywh[1]),
                             (128, 128, 128), 5)
                    
                    
                    

                    cv.rectangle(old_image, (our_xywh[0], our_xywh[1]), (our_xywh[2] + our_xywh[0], our_xywh[3] + our_xywh[1]),
                             (0, 0, 255), 5)
                    cv.rectangle(old_image_our, (our_xywh[0], our_xywh[1]), (our_xywh[2] + our_xywh[0], our_xywh[3] + our_xywh[1]),
                             (128, 128, 128), 5)
                    #original method
                    #history traj
                    cv.rectangle(old_image, (history_xywh[0], history_xywh[1]), (history_xywh[2] + history_xywh[0], history_xywh[3] + history_xywh[1]),
                             (255, 0, 0), 5)
                    cv.rectangle(old_image_hist, (history_xywh[0], history_xywh[1]), (history_xywh[2] + history_xywh[0], history_xywh[3] + history_xywh[1]),
                             (255, 0, 0), 5)
                    
                    
                    


                    cv.imwrite(base_results_path+'/%08d' %frame_index+'neighbors_old'+str(ind)+'.jpg', old_image)
                    cv.imwrite(base_results_path+'/%08d' %frame_index+'neighbors_old_ori'+str(ind)+'.jpg', old_image_ori)
                    cv.imwrite(base_results_path+'/%08d' %frame_index+'neighbors_old_our'+str(ind)+'.jpg', old_image_our)
                    cv.imwrite(base_results_path+'/%08d' %frame_index+'neighbors_old_hist'+str(ind)+'.jpg', old_image_hist)

                #dsa
            
            location = pos_sz_2_xywh(state_neighbor['target_pos'],state_neighbor['target_sz'])
            state = [int(s) for s in location]
            output_boxes.append(state)


            font_color = (0, 0, 0)
            #if 'all_boxes' in out.keys():
            #    neighbor_bbox = np.reshape(out['all_boxes'],(-1,4))
            #else:
            neighbor_bbox = []
            for n_bbox in neighbor_bbox:
                n_state = [int(s) for s in n_bbox]
                print('---have neighbor info---')
                print(neighbor_bbox)
                print(state)

                cv.rectangle(frame_disp, (n_state[0], n_state[1]), (n_state[2] + n_state[0], n_state[3] + n_state[1]),
                         (255,0 , 0), 5)
                
            cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)


            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)

            # Display the resulting frame
            if showcv:
                cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ret, frame = cap.read()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                           (0, 0, 0), 1)
                if showcv:
                    cv.imshow(display_name, frame_disp)
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)
                
            if save_img:
                video_name = Path(videofilepath).stem
                if not os.path.exists(self.results_dir+'/'+video_name):
                    os.makedirs(self.results_dir+'/'+video_name)

                base_results_path = os.path.join(self.results_dir+'/'+video_name)
                cv.imwrite(base_results_path+'/%08d' %frame_index+'.jpg', frame_disp)
                frame_index = frame_index+1


        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')
            
            
