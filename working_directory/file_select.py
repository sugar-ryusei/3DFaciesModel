import glob
import os
import math
import shutil
import random
from natsort import natsorted

#validation dataset ratio relative to the whole dataset
ratio = 0.2

#load labels
label_paths = natsorted(glob.glob('label/*.png'))
#copy labels to training directory
for path in label_paths:
    shutil.copy(path, f'training_data/train/label')
label_paths = [os.path.splitext(os.path.basename(path))[0] for path in label_paths]

#load data path according to the label paths
data_paths = natsorted(glob.glob(f'array/Esashito-PointCloud*/*.npy'))
temp_data_paths = [os.path.splitext(os.path.basename(path))[0] for path in data_paths]
path_bool = [path in label_paths for path in temp_data_paths]
data_paths = [data_paths[i] for i in range(len(data_paths)) if path_bool[i]]
#copy data to training directory
for path in data_paths:
    shutil.copy(path, f'training_data/train/data')

#randomly select validation data and labels
data_paths = natsorted(glob.glob('training_data/train/data/*.npy'))
data_paths = [os.path.splitext(os.path.basename(path))[0] for path in data_paths]
random_sample_path = random.sample(data_paths, math.ceil(len(data_paths)*ratio))

for path in random_sample_path:
    shutil.move(f'training_data/train/data/{path}.npy', f'training_data/test/data')
    shutil.move(f'training_data/train/label/{path}.png', f'training_data/test/label')