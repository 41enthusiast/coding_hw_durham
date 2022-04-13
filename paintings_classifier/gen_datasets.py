from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
from torchvision import transforms, datasets
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import fastai
from fastai.vision import *

import argparse

import os

def download_and_setup_dataset_folder(data_dir):
    path = Path('datasets/source_images')
    classes = []
    for fname in os.listdir(data_dir):
        categ_name = fname.split('_urls')[0]#files should be in this format
        classes.append(categ_name)
        dest = path/categ_name
        dest.mkdir(parents=True, exist_ok=True)
        download_images(fname, dest, max_pics=1000)
        verify_images(path/categ_name, delete=True, max_size=1000)




# draw an image with detected objects
def annotated_images(img, folder_path, result_list, output_mode, output_folder='datasets_anno'):
    # plot the image as base
    img = transforms.ToTensor()(img).permute(1,2,0)
    plt.imshow(img)
    ax = plt.gca()
    # plot each box
    for i,result in enumerate(result_list):
        # get coordinates
        x0, y0, x1, y1 = result
        #print(result)
        rect = Rectangle((x0, y0), x1-x0, y1-y0, fill=False, color='red')
        # draw the box
        ax.add_patch(rect)
    img_path = output_folder+folder_path.split(output_mode)[1]
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')#need this to remove the axes for the final padding
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0, transparent=True)#need this line to save without the extra padding
    plt.show()
    plt.close('all')

#crop out only the faces
def face_crops(img, folder_path, result_list, output_mode, output_folder='datasets_cropped'):
    # plot the image as base
    img = transforms.ToTensor()(img).permute(1,2,0)
    for i,result in enumerate(result_list):
        # get coordinates
        x0, y0, x1, y1 = result
        #print(result)

        #getting the cropped part of the image
        x_coords = int(x0), int(x1)
        y_coords = int(y0), int(y1)
        cropped_img = img[y_coords[0]:y_coords[1], x_coords[0]:x_coords[1]]
        #print(cropped_img.shape)
        cropped_path = +folder_path.split(output_mode)[1].split('.')[0]+'_'+str(i)+'.jpg'
        plt.imsave(cropped_path,cropped_img.numpy())

def gen_MTCNN_processed_images(mtcnn, loader, save_type):
    for i, (x, y) in enumerate(loader):#list of PIL images and image paths
        #print(x,y)
        boxes = mtcnn.detect(x)
        for x_i, y_i, box in zip(x,y, boxes[0]):
            print(x_i.size)
            if type(box) != np.ndarray:#the no faces detected case
                continue
            if save_type == 'annotated':
                annotated_images(x_i, y_i, box, '_output' )
            elif save_type == 'cropped':
                face_crops(x_i, y_i, box, '_output' )
            elif save_type == 'all':
                annotated_images(x_i, y_i, box, '_output' )
                face_crops(x_i, y_i, box, '_output' )
            #print('Image processed')
            
        print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')

def gen_single_face_images(mtcnn, loader, save_type, output_folder='datasets_single_face'):
    for i, (x, y) in enumerate(loader):#list of PIL images and image paths
        #print(x,y)
        boxes = mtcnn.detect(x)
        for x_i, y_i, box in zip(x,y, boxes[0]):
            print(x_i.size)
            if type(box) != np.ndarray:#the no faces detected case
                continue
            if len(box)==1:#if only a single face has been detected
                if save_type == 'annotated':
                    annotated_images(x_i, y_i, box, '_single' )
                elif save_type == 'cropped':
                    face_crops(x_i, y_i, box, '_single' )
                elif save_type == 'all':
                    annotated_images(x_i, y_i, box, '_single' )
                    face_crops(x_i, y_i, box )
                    plt.imsave(output_folder+y_i.split('_single')[1], x_i)
                else:
                    plt.imsave(output_folder+y_i.split('_single')[1], x_i)
            #print('Image processed')
            
        print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')

def setup_ds_dl(args, output_mode):
    data_dir = args.data_dir+'_'+args.mode
    img_size = args.img_size

    #make the parent directories
    os.makedirs(data_dir,exist_ok=True)

    #set up the dataset and dataloaders for the input data directory
    temp_transforms = transforms.Compose([transforms.Resize(img_size)])

    dataset = datasets.ImageFolder(data_dir, transform= temp_transforms)
    dataset.samples = [
        (p, p.replace(data_dir, data_dir + output_mode))
            for p, _ in dataset.samples
    ]
            
    loader = DataLoader(
        dataset,
        num_workers=args.n_workers,
        batch_size=args.batch_size,
        collate_fn=training.collate_pil
    )

    #code to make the child directories
    for categ in dataset.classes:
        os.makedir(data_dir+'/'+categ)

    return dataset, loader

#input dataset dir -> extract the classes from it and use it for the target_directories
if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Paintings Classifier')

    parser.add_argument('--image-size', type=int, default=512, metavar='N',
                        help='input image size for model training and inference (default: 512)')

    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                        help='Number of workers (default: 4)')

    

    parser.add_argument('--data-dir', type=str, default = 'datasets/datasets',
                        help='Path to directory with imagenet style dataset or with csv files of categories with image links')
    parser.add_argument('--output-dir', type=str, default = 'datasets/datasets_multiple',
                        help="Name of the dataset path for the MTCNN processed outputs")
    parser.add_argument('--output-single-dir', type=str, default = 'datasets/single_faces',
                        help="Name of the dataset path for the single face cropped images outputs")
    
    parser.add_argument('--mode', type=str, choices=['annotated','all','cropped','plain'], default='cropped',
                        help="Type of processed face images (options: 'cropped' for only face cropped images, 'annotated' for bounding boxes on detected faces,'plain' for no MTCNN processing ,'all' for all the options)")
    parser.add_argument('--output-mode', type=str, choices=['multiple','single'], default='single',
                        help="Process images with 'multiple' or 'single' faces?")
    parser.add_argument('--input-mode', type=str, choices=['existing','links'], default='existing',
                        help="Type of input face dataset directory (options: 'links' to download images from a csv file with urls, 'existing': imagenet dataset style folders)")#can include the option to extract the tar.gz file

    args = parser.parse_args()

    if args.input_mode=='links':
        download_and_setup_dataset_folder(args.data_dir)

    if args.output_mode == 'multiple':
        output_mode = '_output'
    else:
        output_mode = '_single'

    dataset, loader = setup_ds_dl(args, output_mode)
    
    #setting up the MTCNN model here
    mtcnn = MTCNN(post_process=False, keep_all = True)

    if output_mode == '_output':
        gen_MTCNN_processed_images(mtcnn, loader, args.mode)
    else:
        gen_single_face_images(mtcnn, loader, args.mode)

