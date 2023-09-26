# [Memory Replay with Data Compression for Continual Learning (ICLR 2022)]() 

------
This code is the official implementation of our paper.

In this work, we propose memory replay with data compression (MRDC), which can largely reduce the storage of old training samples and thus increase their amount that can be stored in the memory buffer. 
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
For LUCIR w/ MRDC, the execution is similar to that of LUCIR:

- cd `2_LUCIR_+DC`

- see `run.sh` for the experiments on ImageNet-sub

- see `run_all.sh` for the experiments on ImageNet-full

- To determine the quality with our method described in Sec.4.2, please load the checkpoint and run `class_incremental_cosine_imagenet_Rq.py`


Compared with LUCIR, you need to additionally edit:

--traindir_compression for the data path of compressed data; 
--quality for the quality; and --nb_protos for the quantity, which is computed by the compression rate given the quality (detailed in `1_Data_Compression/imagenet_jpeg.sh`).



## **3. Expected Results**

Here we provide the phase-wise results (with buffer size of 20 images / class) to calculate the averaged incremental accuracy in our paper. The results might slightly vary due to different random seeds.  

(1) ImageNet-sub																											
LUCIR w/ MRDC 5-phase:		83.72	78.35	74.37	71.21	68.19	65.57																				
LUCIR w/ MRDC 10-phase:		83.37	80.46	77.68	75.50	73.45	71.96	70.90	69.42	67.99	66.69	64.99															
LUCIR w/ MRDC 25-phase:		83.91	82.33	80.50	79.46	78.53	77.15	76.09	74.94	73.90	73.13	72.43	72.08	71.04	70.38	69.91	69.30	68.69	67.97	67.28	67.00	66.46	65.73	65.08	64.55	64.07	63.72

(2) ImageNet-full   
LUCIR w/ MRDC 5-phase:		75.94	71.66	67.92	65.5	63.21	61.23					
LUCIR w/ MRDC 10-phase:		75.94	72.6	70.48	68.24	66.22	64.67	62.91	61.26	60.01	58.41	57.55


## **Citation**

Please cite our paper if it is helpful to your work:

```bibtex
@inproceedings{wang2021memory,
  title={Memory Replay with Data Compression for Continual Learning},
  author={Wang, Liyuan and Zhang, Xingxing and Yang, Kuo and Yu, Longhui and Li, Chongxuan and Lanqing, HONG and Zhang, Shifeng and Li, Zhenguo and Zhong, Yi and Zhu, Jun},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```
