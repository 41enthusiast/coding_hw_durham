import shutil
from glob import glob
import os

data_path = '../../DURHAM_EXPORTS-20220323T162326Z-001/DURHAM_EXPORTS'
folders = ['Shakespeare', 'Queen_Victoria']

for folder in folders:
    os.mkdir('datasets/'+folder)
    source = data_path+'/'+folder+'/media/*.jpg'
    for fname in glob(source):
        shutil.copyfile(fname, 'datasets/'+folder+'/'+fname.split('/')[-1])