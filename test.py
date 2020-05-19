import os
os.environ["DLClight"]="True"
import deeplabcut

path_config_file = '/home/srinidogo/DeepLabCut/examples/cloned-DLC-repo/cloned-DLC-repo/examples/openfield-Pranav-2018-10-30/config.yaml'
deeplabcut.load_demo_data(path_config_file)
deeplabcut.train_network(path_config_file, shuffle=1, displayiters=10,saveiters=100)
videofile_path = ['/home/srinidogo/DeepLabCut/examples/cloned-DLC-repo/cloned-DLC-repo/examples/openfield-Pranav-2018-10-30/videos/m3v1mp4.mp4']
deeplabcut.analyze_videos(path_config_file,videofile_path, videotype='.mp4')
deeplabcut.create_labeled_video(path_config_file,videofile_path)
deeplabcut.plot_trajectories(path_config_file,videofile_path)

