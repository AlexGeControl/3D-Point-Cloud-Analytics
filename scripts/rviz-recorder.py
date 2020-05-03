import glob
import os
from xml.dom import minidom
import pandas as pd

import docker
import signal
import subprocess

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

# set input measurement:
MEASUREMENT = '2019-11-22_CJ-006_2019.47_ROUND_V716216_QianShan_Integration_FT-Zett/2019-11-22_19-45-15'
OFFSET = 2.5
DURATION = 10.0

def parse_defect_markers(measurement):
    """ parse markers as defect dataframe
    """
    # identify marker file:
    pattern = os.path.join(
        "/home/bmw/data/rosbag/ROS_BAG/Overall_performance",
        MEASUREMENT,
        "*_MARKER.xml"
    )

    filename = glob.glob(
        pattern
    )[0]

    # parse defects:
    defects = [
        {
            "filename": marker.attributes['filename'].value,
            "walltime_in_sec": marker.attributes['walltime_in_sec'].value,
            "bagtime_in_sec": marker.attributes['bagtime_in_sec'].value,
            "description": marker.attributes['description'].value,
        }   for marker in minidom.parse(filename).getElementsByTagName('Marker') 
    ]

    df_defects = pd.DataFrame.from_dict(
        {
            k: [defect[k] for defect in defects] for k in defects[0].keys()
        }
    )

    return df_defects

def get_rviz_video(MEASUREMENT, ros_bag_name):
    """ get rviz video
    """
    # get docker client:
    client = docker.from_env()

    # init VNC recorder:
    proc_vnc_recorder = subprocess.Popen(
        ['flvrec.py', 'localhost', '35901']
    )

    # launch rosbag play:
    bmw_ros_instance = client.containers.list()[0]
    cmd_rosbag_play = '/bin/bash -c "source /workspace/bazel/setup.sh ; rosbag play --clock /data/rosbag/{}"'.format(
        os.path.join(MEASUREMENT, ros_bag_name)
    )
    log = bmw_ros_instance.exec_run(
        cmd_rosbag_play,
        stderr=True,
        stdout=True
    )

    # stop VNC recorder:
    proc_vnc_recorder.send_signal(signal.SIGINT)
    proc_vnc_recorder.wait()

# get defects:
df_defects = parse_defect_markers(MEASUREMENT)

for ros_bag_name, defects in df_defects.groupby('filename'):
    # generate RViz video:
    get_rviz_video(MEASUREMENT, ros_bag_name)
    rviz_video_name = sorted(glob.glob("*.flv"))[-1]
    # cut for each defect:
    for index, defect in defects.iterrows():
        # defect timestamp:
        timestamp_center = float(defect['bagtime_in_sec']) + OFFSET
        (timestamp_begin, timestamp_end) = (timestamp_center - DURATION / 2.0, timestamp_center + DURATION / 2.0)
        # defect video name:
        defect_video_name = "{:02d}_{}_{}.flv".format(
            index,
            "{}@{}".format(ros_bag_name, defect['walltime_in_sec']), 
            "-".join(defect['description'].split())
        )
        ffmpeg_extract_subclip(
            rviz_video_name, 
            timestamp_begin, timestamp_end, 
            targetname=defect_video_name
        )
    # continue:
    raw_input("Reset RViz then press Enter to continue...")
exit(0)
