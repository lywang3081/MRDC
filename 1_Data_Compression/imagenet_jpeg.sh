# Step 1
# Get folder list
imagenet256_dir="/datapath/seed_1993_subset_100_imagenet"  #replace this with your data path
find $imagenet256_dir -mindepth 2 -type d > subfolder_list.txt

# Step 2
# Preprocess data
quality=10   #replace this with your quality
i=1
for line in $(cat subfolder_list.txt)
do
  #if(($i==2))
  #then
  #break
  #fi

  echo $i
  input_dir=$line
  output_dir=${line/seed_1993_subset_100_imagenet/"seed_1993_subset_100_imagenet_quality_"$quality}
  echo $input_dir
  echo $output_dir

  python skimage_save_jpeg.py \
    --input_dir $input_dir \
    --output_dir $output_dir \
    --quality $quality

  ((++i))
done

# Step 3
# Compute the compression rate (manually)
#cd your_data_path
#du -h --max-depth 1
