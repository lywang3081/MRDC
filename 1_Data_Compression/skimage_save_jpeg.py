import glob
import os
import argparse
from tqdm import tqdm
from skimage import io

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='datapath/JPEG-quality-imagenet256/n01440764/')
    parser.add_argument('--output_dir', default='datapath/JPEG-quality-imagenet256/n01440764_Q=5/')
    parser.add_argument('--quality', default=5)
    args = parser.parse_args()
    print(args)
    input_dir = args.input_dir if args.input_dir.endswith('/') else args.input_dir + '/'
    output_dir = args.output_dir if args.output_dir.endswith('/') else args.output_dir + '/'
    return input_dir, output_dir, args.quality


def get_JPEG_img(img_dir):
    path_list = glob.glob(img_dir+'*.JPEG')
    print('Image directory: ', img_dir)
    print('Number of JPEG files: ', len(path_list))
    return path_list


img_dir, img_dir_new, quality = get_args()

img_path_list = get_JPEG_img(img_dir)

if not os.path.isdir(img_dir_new):
    os.makedirs(img_dir_new)
print('New image directory: ', img_dir_new)

for img_path in tqdm(img_path_list):
    image = io.imread(fname=img_path)
    img_path_new = img_path.replace(img_dir, img_dir_new).replace('.JPEG', f"_Q={quality}.JPEG")
    # print(img_path_new)
    # exit()
    io.imsave(fname=img_path_new, arr=image, quality=quality)