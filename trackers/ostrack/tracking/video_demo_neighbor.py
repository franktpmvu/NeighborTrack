import os
import sys
import argparse
sys.path.append('/data/')
from NeighborTrack.neighbortrack import neighbortrack
from NeighborTrack.NTutils.utils import xy_wh_2_rect

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import Tracker,NeighborTracker


def run_video(tracker_name, tracker_param, videofile, optional_box=None, debug=None, save_results=False,save_img=False ):
    """Run the tracker on your webcam.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        debug: Debug level.
    """
    tracker = NeighborTracker(tracker_name, tracker_param, "video")
    invTracker = Tracker(tracker_name, tracker_param, "video")
    tracker.run_video_neighbor(videofilepath=videofile, optional_box=optional_box, debug=debug, save_results=save_results,save_img=save_img ,revtracker=invTracker)


def main():
    parser = argparse.ArgumentParser(description='Run the tracker on your webcam.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
    parser.add_argument('videofile', type=str, help='path to a video file.')
    parser.add_argument('--optional_box', type=float, default=None, nargs="+", help='optional_box with format x y w h.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--save_results', dest='save_results', action='store_true', help='Save bounding boxes')
    parser.add_argument('--save_img', dest='save_img', action='store_true', help='Save bounding boxes image')

    parser.set_defaults(save_results=False)

    args = parser.parse_args()

    run_video(args.tracker_name, args.tracker_param, args.videofile, args.optional_box, args.debug, args.save_results, args.save_img)


if __name__ == '__main__':
    main()
