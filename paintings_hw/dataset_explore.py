import pandas as pd
from glob import glob
data_path = '../../DURHAM_EXPORTS-20220323T162326Z-001/DURHAM_EXPORTS'
folders = ['Shakespeare', 'Queen_Victoria']

fnames_su = ['shakespeare', 'victoria']
fnames_tags = ['_media.csv', '_tags.csv']


#media file: 'id' 'name' 'pcf_accession_number' 'id.1' 'filename' 'path'
#tags file: 'id' 'name' 'pcf_accession_number' 'id.1' 'elastic_label' 'Term ID' 'Term Name' 'Person ID' 'Term Name.1'
for i,folder in enumerate(folders):
    print('Folder: ', folder)
    folder_path = data_path+'/'+folder+'/'
    for fname in fnames_tags:
        print('file:', fnames_su[i]+fname)
        csv_file = pd.read_csv(folder_path+fnames_su[i]+fname)
        print(csv_file)
        print('Column heads: ')
        print(csv_file.columns.values) 
        print()

