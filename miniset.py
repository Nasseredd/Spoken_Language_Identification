import os
import glob
import random
import shutil

dataset = os.path.normpath('C:\\Users\\user\\Desktop\\train\\train')
miniset = os.path.normpath('C:\\Users\\user\\Desktop\\train\\mini\\')

files = glob.glob(dataset + os.sep + "*.flac")
sample = random.sample(files, 1000)

[shutil.copy(file, miniset) for file in sample]