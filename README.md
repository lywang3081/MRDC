
# [Memory Replay with Data Compression for Continual Learning]() 

------
This code is the official implementation of our paper.

In this work, we propose memory replay with data compression, which can largely reduce the storage of old training samples and thus increase their amount that can be stored in the memory buffer. 
Our work can be easily implemented into representative memory replay approaches.
Here we provide the implementation on LUCIR as an example.


## **Dependencies**
- Python 3.6 (Anaconda3 Recommended)
- Pytorch 0.4.0

## **1. Prepare Compressed Data**
Your need to first prepare the compressed dataset:

- cd `1_Data_Compression`

- run `get_folders.sh` with your data path

- run `imagenet_jpeg.sh` with your data path and quality

## **2. Conduct Experiments**
For LUCIR w/ ours, the execution is similar to that of LUCIR:

- cd `2_LUCIR_+DC`

- see `run.sh` for the experiments on ImageNet-sub

- see `run_all.sh` for the experiments on ImageNet-full

- To determine the quality with our method described in Sec.4.2, please load the checkpoint and run `class_incremental_cosine_imagenet_Rq.py`


Compared with LUCIR, you need to additionally edit:

--traindir_compression for the data path of compressed data; 
--quality for the quality; and --nb_protos for the quantity, which is computed by the compression rate given the quality (detailed in `1_Data_Compression/imagenet_jpeg.sh`).



## **Citation**

Please cite our paper if it is helpful to your work:

```bibtex
@article{wang2022memory,
  title={Memory Replay with Data Compression for Continual Learning},
  author={Wang, Liyuan and Zhang, Xingxing and Yang, Kuo and Yu, Longhui and Li, Chongxuan and Hong, Lanqing and Zhang, Shifeng and Li, Zhenguo and Zhong, Yi and Zhu, Jun},
  journal={arXiv preprint arXiv:2202.06592},
  year={2022}
}
```
