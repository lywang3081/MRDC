# Step 1
# List all directories and subdirectories
find /datapath/ImageNet -type d > folder_list.txt    #replace this with your data path

# Step 2
# Remove lines manually (top directories: current directory, train and test)
# Save as subfolder_list.txt