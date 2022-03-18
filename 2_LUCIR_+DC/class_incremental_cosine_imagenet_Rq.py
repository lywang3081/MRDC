#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import numpy as np
import time
import os
import sys
import copy
import argparse
from PIL import Image
try:
    import cPickle as pickle
except:
    import pickle
import math

import modified_resnet
import modified_linear
import utils_pytorch
from utils_imagenet.utils_dataset import split_images_labels
from utils_imagenet.utils_dataset import merge_images_labels
from utils_incremental.compute_features import compute_features
from utils_incremental.compute_accuracy import compute_accuracy
from utils_incremental.compute_confusion_matrix import compute_confusion_matrix
from utils_incremental.incremental_train_and_eval import incremental_train_and_eval
from utils_incremental.incremental_train_and_eval_MS import incremental_train_and_eval_MS
from utils_incremental.incremental_train_and_eval_LF import incremental_train_and_eval_LF
from utils_incremental.incremental_train_and_eval_MR_LF import incremental_train_and_eval_MR_LF
from utils_incremental.incremental_train_and_eval_AMR_LF import incremental_train_and_eval_AMR_LF

import time

######### Modifiable Settings ##########
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='seed_1993_subset_100_imagenet', type=str)
parser.add_argument('--num_classes', default=100, type=int)
parser.add_argument('--num_workers', default=16, type=int, \
    help='the number of workers for loading data')
parser.add_argument('--nb_cl_fg', default=50, type=int, \
    help='the number of classes in first group')
parser.add_argument('--nb_cl', default=10, type=int, \
    help='Classes per group')
parser.add_argument('--nb_protos', default=20, type=int, \
    help='Number of prototypes per class at the end')
parser.add_argument('--nb_runs', default=1, type=int, \
    help='Number of runs (random ordering of classes at each run)')
parser.add_argument('--ckp_prefix', default=os.path.basename(sys.argv[0])[:-3], type=str, \
    help='Checkpoint prefix')
parser.add_argument('--epochs', default=90, type=int, \
    help='Epochs')
parser.add_argument('--T', default=2, type=float, \
    help='Temporature for distialltion')
parser.add_argument('--beta', default=0.25, type=float, \
    help='Beta for distialltion')
parser.add_argument('--resume', action='store_true', \
    help='resume from checkpoint')
parser.add_argument('--fix_budget', action='store_true', \
    help='fix budget')
########################################
parser.add_argument('--mimic_score', action='store_true', \
    help='To mimic scores for cosine embedding')
parser.add_argument('--lw_ms', default=1, type=float, \
    help='loss weight for mimicking score')
########################################
#improved class incremental learning
parser.add_argument('--rs_ratio', default=0, type=float, \
    help='The ratio for resample')
parser.add_argument('--imprint_weights', action='store_true', \
    help='Imprint the weights for novel classes')
parser.add_argument('--less_forget', action='store_true', \
    help='Less forgetful')
parser.add_argument('--lamda', default=5, type=float, \
    help='Lamda for LF')
parser.add_argument('--adapt_lamda', action='store_true', \
    help='Adaptively change lamda')
parser.add_argument('--mr_loss', action='store_true', \
    help='Margin ranking loss v1')
parser.add_argument('--amr_loss', action='store_true', \
    help='Margin ranking loss v2')
parser.add_argument('--dist', default=0.5, type=float, \
    help='Dist for MarginRankingLoss')
parser.add_argument('--K', default=2, type=int, \
    help='K for MarginRankingLoss')
parser.add_argument('--lw_mr', default=1, type=float, \
    help='loss weight for margin ranking loss')
########################################
parser.add_argument('--random_seed', default=1993, type=int, \
    help='random seed')
parser.add_argument('--datadir', default='/home/username/data/ImageNet/seed_1993_subset_100_imagenet/', type=str)
parser.add_argument('--traindir_compression', default='/home/username/data/ImageNet/seed_1993_subset_100_imagenet_quality_5/train', type=str)
parser.add_argument('--quality', default=10, type=float)
args = parser.parse_args()

########################################
assert(args.nb_cl_fg % args.nb_cl == 0)
assert(args.nb_cl_fg >= args.nb_cl)
train_batch_size       = 128            # Batch size for train
test_batch_size        = 50             # Batch size for test
eval_batch_size        = 128            # Batch size for eval
base_lr                = 0.1            # Initial learning rate
lr_strat               = [30, 60]       # Epochs where learning rate gets decreased
lr_factor              = 0.1            # Learning rate decrease factor
custom_weight_decay    = 1e-4           # Weight Decay
custom_momentum        = 0.9            # Momentum

args.ckp_prefix        = 'class_incremental_cosine_imagenet_V2'
#args.ckp_prefix        = '{}_nb_cl_fg_{}_nb_cl_{}_nb_protos_{}_quality_{}'.format(args.ckp_prefix, args.nb_cl_fg, args.nb_cl, args.nb_protos, args.quality)
#args.ckp_prefix        = '{}_nb_cl_fg_{}_nb_cl_{}_nb_protos_{}_quality_{}_seed_{}'.format(args.ckp_prefix, args.nb_cl_fg, args.nb_cl, args.nb_protos, args.quality, args.random_seed)
args.ckp_prefix        = '{}_nb_cl_fg_{}_nb_cl_{}_nb_protos_{}_quality_{}_seed_{}'.format(args.ckp_prefix, args.nb_cl_fg, args.nb_cl, 20, 100.0, args.random_seed)
np.random.seed(args.random_seed)        # Fix the random seed
print(args)
########################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
traindir = args.datadir + 'train'
valdir = args.datadir + 'val'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
trainset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
testset =  datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
evalset =  datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

# load compressed data
transform_train_compression = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,])

traindir_compression = args.traindir_compression
trainset_compression = datasets.ImageFolder(traindir_compression, transform_train_compression)

# Initialization
dictionary_size     = 1500
top1_acc_list_cumul = np.zeros((int(args.num_classes/args.nb_cl),3,args.nb_runs))
top1_acc_list_ori   = np.zeros((int(args.num_classes/args.nb_cl),3,args.nb_runs))

X_train_total, Y_train_total = split_images_labels(trainset.imgs)
X_valid_total, Y_valid_total = split_images_labels(testset.imgs)
X_train_total_compression, Y_train_total_compression = split_images_labels(trainset_compression.imgs)

start = time.clock()
# Launch the different runs
for iteration_total in range(args.nb_runs):
    # Select the order for the class learning
    #order_name = "./checkpoint_LUCIR/seed_{}_{}_order_run_{}.pkl".format(args.random_seed, args.dataset, iteration_total)
    order_name = "./checkpoint/seed_{}_{}_order_run_{}.pkl".format(args.random_seed, args.dataset, iteration_total)
    print("Order name:{}".format(order_name))
    if os.path.exists(order_name):
        print("Loading orders")
        order = utils_pytorch.unpickle(order_name)
    else:
        print("Generating orders")
        order = np.arange(args.num_classes)
        np.random.shuffle(order)
        utils_pytorch.savepickle(order, order_name)
    order_list = list(order)
    print(order_list)

    # Initialization of the variables for this run
    X_valid_cumuls    = []
    X_protoset_cumuls = []
    X_train_cumuls    = []
    Y_valid_cumuls    = []
    Y_protoset_cumuls = []
    Y_train_cumuls    = []
    alpha_dr_herding  = np.zeros((int(args.num_classes/args.nb_cl),dictionary_size,args.nb_cl),np.float32)

    # The following contains all the training samples of the different classes
    # because we want to compare our method with the theoretical case where all the training samples are stored
    # prototypes = np.zeros((args.num_classes,dictionary_size,X_train_total.shape[1],X_train_total.shape[2],X_train_total.shape[3]))
    prototypes = [[] for i in range(args.num_classes)]
    for orde in range(args.num_classes):
        prototypes[orde] = X_train_total_compression[np.where(Y_train_total_compression==order[orde])]
    prototypes = np.array(prototypes)

    start_iter = int(args.nb_cl_fg/args.nb_cl)-1
    for iteration in range(start_iter, int(args.num_classes/args.nb_cl)): #continual learning of tasks
        #init model
        if iteration == start_iter:
            ############################################################
            last_iter = 0
            ############################################################
            tg_model = modified_resnet.resnet18(num_classes=args.nb_cl_fg)
            in_features = tg_model.fc.in_features
            out_features = tg_model.fc.out_features
            print("in_features:", in_features, "out_features:", out_features)
            ref_model = None
        elif iteration == start_iter+1:
            ############################################################
            last_iter = iteration
            ############################################################
            #increment classes
            ref_model = copy.deepcopy(tg_model)
            in_features = tg_model.fc.in_features
            out_features = tg_model.fc.out_features
            print("in_features:", in_features, "out_features:", out_features)
            new_fc = modified_linear.SplitCosineLinear(in_features, out_features, args.nb_cl)
            new_fc.fc1.weight.data = tg_model.fc.weight.data
            new_fc.sigma.data = tg_model.fc.sigma.data
            tg_model.fc = new_fc
            lamda_mult = out_features*1.0 / args.nb_cl
        else:
            ############################################################
            last_iter = iteration
            ############################################################
            ref_model = copy.deepcopy(tg_model)
            in_features = tg_model.fc.in_features
            out_features1 = tg_model.fc.fc1.out_features
            out_features2 = tg_model.fc.fc2.out_features
            print("in_features:", in_features, "out_features1:", \
                out_features1, "out_features2:", out_features2)
            new_fc = modified_linear.SplitCosineLinear(in_features, out_features1+out_features2, args.nb_cl)
            new_fc.fc1.weight.data[:out_features1] = tg_model.fc.fc1.weight.data
            new_fc.fc1.weight.data[out_features1:] = tg_model.fc.fc2.weight.data
            new_fc.sigma.data = tg_model.fc.sigma.data
            tg_model.fc = new_fc
            lamda_mult = (out_features1+out_features2)*1.0 / (args.nb_cl)

        if iteration > start_iter and args.less_forget and args.adapt_lamda:
            #cur_lamda = lamda_base * sqrt(num_old_classes/num_new_classes)
            cur_lamda = args.lamda * math.sqrt(lamda_mult)
        else:
            cur_lamda = args.lamda
        if iteration > start_iter and args.less_forget:
            print("###############################")
            print("Lamda for less forget is set to ", cur_lamda)
            print("###############################")

        # Prepare the training data for the current batch of classes
        actual_cl        = order[range(last_iter*args.nb_cl,(iteration+1)*args.nb_cl)]
        indices_train_10 = np.array([i in order[range(last_iter*args.nb_cl,(iteration+1)*args.nb_cl)] for i in Y_train_total])
        indices_test_10  = np.array([i in order[range(last_iter*args.nb_cl,(iteration+1)*args.nb_cl)] for i in Y_valid_total])

        X_train          = X_train_total[indices_train_10]  #training data of the new task
        X_valid          = X_valid_total[indices_test_10]
        X_valid_cumuls.append(X_valid)
        X_train_cumuls.append(X_train)
        X_valid_cumul    = np.concatenate(X_valid_cumuls)
        X_train_cumul    = np.concatenate(X_train_cumuls)

        Y_train          = Y_train_total[indices_train_10]
        Y_valid          = Y_valid_total[indices_test_10]
        Y_valid_cumuls.append(Y_valid)
        Y_train_cumuls.append(Y_train)
        Y_valid_cumul    = np.concatenate(Y_valid_cumuls)
        Y_train_cumul    = np.concatenate(Y_train_cumuls)

        X_train_ori = X_train
        Y_train_ori = Y_train

        # Add the stored exemplars to the training data
        if iteration == start_iter:
            X_valid_ori = X_valid
            Y_valid_ori = Y_valid
        else:
            X_protoset = np.concatenate(X_protoset_cumuls) #cat all replay data
            Y_protoset = np.concatenate(Y_protoset_cumuls)
            if args.rs_ratio > 0:
                scale_factor = (len(X_train) * args.rs_ratio) / (len(X_protoset) * (1 - args.rs_ratio))
                rs_sample_weights = np.concatenate((np.ones(len(X_train)), np.ones(len(X_protoset))*scale_factor))
                rs_num_samples = int(len(X_train) / (1 - args.rs_ratio))
                print("X_train:{}, X_protoset:{}, rs_num_samples:{}".format(len(X_train), len(X_protoset), rs_num_samples))
            X_train    = np.concatenate((X_train,X_protoset),axis=0) #training data of the new task + all replay data
            Y_train    = np.concatenate((Y_train,Y_protoset))

        # Launch the training loop
        print('Batch of classes number {0} arrives ...'.format(iteration+1))
        map_Y_train = np.array([order_list.index(i) for i in Y_train])
        map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])

        #imprint weights
        if iteration > start_iter and args.imprint_weights:
            print("Imprint weights")
            #########################################
            #compute the average norm of old embdding
            old_embedding_norm = tg_model.fc.fc1.weight.data.norm(dim=1, keepdim=True)
            average_old_embedding_norm = torch.mean(old_embedding_norm, dim=0).to('cpu').type(torch.DoubleTensor)
            #########################################
            tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])
            num_features = tg_model.fc.in_features
            novel_embedding = torch.zeros((args.nb_cl, num_features))
            for cls_idx in range(iteration*args.nb_cl, (iteration+1)*args.nb_cl):
                cls_indices = np.array([i == cls_idx  for i in map_Y_train])
                assert(len(np.where(cls_indices==1)[0])<=dictionary_size)
                current_eval_set = merge_images_labels(X_train[cls_indices], np.zeros(len(X_train[cls_indices])))
                evalset.imgs = evalset.samples = current_eval_set
                evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size, shuffle=False, num_workers=2)
                num_samples = len(X_train[cls_indices])
                cls_features = compute_features(tg_feature_model, evalloader, num_samples, num_features)
                norm_features = F.normalize(torch.from_numpy(cls_features), p=2, dim=1)
                cls_embedding = torch.mean(norm_features, dim=0)
                novel_embedding[cls_idx-iteration*args.nb_cl] = F.normalize(cls_embedding, p=2, dim=0) * average_old_embedding_norm
            tg_model.to(device)
            tg_model.fc.fc2.weight.data = novel_embedding.to(device)

        ############################################################
        current_train_imgs = merge_images_labels(X_train, map_Y_train)
        trainset.imgs = trainset.samples = current_train_imgs
        if iteration > start_iter and args.rs_ratio > 0 and scale_factor > 1:
            print("Weights from sampling:", rs_sample_weights)
            index1 = np.where(rs_sample_weights>1)[0]
            index2 = np.where(map_Y_train<iteration*args.nb_cl)[0]
            assert((index1==index2).all())
            train_sampler = torch.utils.data.sampler.WeightedRandomSampler(rs_sample_weights, rs_num_samples)
            #trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, \
            #    shuffle=False, sampler=train_sampler, num_workers=2)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, \
                shuffle=False, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)             
        else:
            #trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
            #    shuffle=True, num_workers=2)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                shuffle=True, num_workers=args.num_workers, pin_memory=True)
        #testset.test_data = X_valid_cumul.astype('uint8')
        #testset.test_labels = map_Y_valid_cumul
        current_test_imgs = merge_images_labels(X_valid_cumul, map_Y_valid_cumul)
        testset.imgs = testset.samples = current_test_imgs
        testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
            shuffle=False, num_workers=2)
        print('Max and Min of train labels: {}, {}'.format(min(map_Y_train), max(map_Y_train)))
        print('Max and Min of valid labels: {}, {}'.format(min(map_Y_valid_cumul), max(map_Y_valid_cumul)))
        ##############################################################
        #ckp_name = './checkpoint_LUCIR/{}_run_{}_iteration_{}_model.pth'.format(args.ckp_prefix, iteration_total, iteration)
        ckp_name = './checkpoint/{}_run_{}_iteration_{}_model.pth'.format(args.ckp_prefix, iteration_total, iteration)
        print('ckp_name', ckp_name)
        #Always load the pretrained model for ploting the coverage
        if args.resume and os.path.exists(ckp_name):
            print("###############################")
            print("Loading models from checkpoint")
            tg_model = torch.load(ckp_name)
            print("###############################")

            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')

            if not os.path.isdir('checkpoint_feature'):
                os.mkdir('checkpoint_feature')

        ### Using compression data to select prototype (with feature mean)
        ### Exemplars
        if args.fix_budget:
            nb_protos_cl = int(np.ceil(args.nb_protos*args.num_classes*1.0/args.nb_cl/(iteration+1)))
            print("fixed nb_protos_cl", nb_protos_cl)
        else:
            nb_protos_cl = args.nb_protos
            print("no fixed nb_protos_cl", nb_protos_cl)

        tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])
        num_features = tg_model.fc.in_features

        # Herding
        print('Updating exemplar set...')
        for iter_dico in range(last_iter*args.nb_cl, (iteration+1)*args.nb_cl):
            current_eval_set = merge_images_labels(prototypes[iter_dico], np.zeros(len(prototypes[iter_dico])))
            evalset.imgs = evalset.samples = current_eval_set
            evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            num_samples = len(prototypes[iter_dico])
            mapped_prototypes = compute_features(tg_feature_model, evalloader, num_samples, num_features) # compute feature, with dataloader
            D = mapped_prototypes.T
            D = D/np.linalg.norm(D,axis=0)

            # Herding procedure : ranking of the potential exemplars
            mu  = np.mean(D,axis=1)
            index1 = int(iter_dico/args.nb_cl)
            index2 = iter_dico % args.nb_cl
            alpha_dr_herding[index1,:,index2] = alpha_dr_herding[index1,:,index2]*0
            w_t = mu
            iter_herding     = 0
            iter_herding_eff = 0
            while not(np.sum(alpha_dr_herding[index1,:,index2]!=0)==min(nb_protos_cl,500)) and iter_herding_eff<1000:
                tmp_t   = np.dot(w_t,D)
                ind_max = np.argmax(tmp_t)
                iter_herding_eff += 1
                if alpha_dr_herding[index1,ind_max,index2] == 0:
                    alpha_dr_herding[index1,ind_max,index2] = 1+iter_herding
                    iter_herding += 1
                w_t = w_t+mu-D[:,ind_max]

        # Prepare the protoset
        X_protoset_cumuls = []
        Y_protoset_cumuls = []

        # Class means for iCaRL and NCM + Storing the selected exemplars in the protoset
        print('Computing mean-of_exemplars and theoretical mean...')
        # class_means = np.zeros((64,100,2))
        class_means = np.zeros((num_features, args.num_classes, 2))

        volume_all_cls = 0
        num_all_cls = 1e-9
        for iteration2 in range(iteration+1):
            #print("Current Task is", iteration2)
            for iter_dico in range(args.nb_cl):
                current_cl = order[range(iteration2*args.nb_cl,(iteration2+1)*args.nb_cl)]
                # Collect data in the feature space for each class
                current_eval_set = merge_images_labels(prototypes[iteration2*args.nb_cl+iter_dico],  np.zeros(len(prototypes[iteration2*args.nb_cl+iter_dico]))) # for each class
                evalset.imgs = evalset.samples = current_eval_set
                evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
                num_samples = len(prototypes[iteration2*args.nb_cl+iter_dico])
                mapped_prototypes = compute_features(tg_feature_model, evalloader, num_samples, num_features) # compute feature, with dataloader
                D = mapped_prototypes.T
                D = D/np.linalg.norm(D,axis=0)
                D2 = D

                # iCaRL
                alph = alpha_dr_herding[iteration2,:,iter_dico]
                assert((alph[num_samples:]==0).all())
                alph = alph[:num_samples]
                alph = (alph>0)*(alph<nb_protos_cl+1)*1.
                # X_protoset_cumuls.append(prototypes[iteration2*args.nb_cl+iter_dico,np.where(alph==1)[0]])
                X_protoset_cumuls.append(prototypes[iteration2*args.nb_cl+iter_dico][np.where(alph==1)[0]])
                Y_protoset_cumuls.append(order[iteration2*args.nb_cl+iter_dico]*np.ones(len(np.where(alph==1)[0])))

                # selected prototype of each class
                X_protoset_current = prototypes[iteration2*args.nb_cl+iter_dico][np.where(alph==1)[0]]
                Y_protoset_current = order[iteration2*args.nb_cl+iter_dico]*np.ones(len(np.where(alph==1)[0]))
                #print('Y_protoset_current', Y_protoset_current)

                alph = alph/np.sum(alph)
                class_means[:,current_cl[iter_dico],0] = (np.dot(D,alph)+np.dot(D2,alph))/2
                class_means[:,current_cl[iter_dico],0] /= np.linalg.norm(class_means[:,current_cl[iter_dico],0])

                # Normal NCM
                # alph = np.ones(dictionary_size)/dictionary_size
                alph = np.ones(num_samples)/num_samples
                class_means[:,current_cl[iter_dico],1] = (np.dot(D,alph)+np.dot(D2,alph))/2
                class_means[:,current_cl[iter_dico],1] /= np.linalg.norm(class_means[:,current_cl[iter_dico],1])

                if iteration2 >= last_iter:
                    #print("last_iter", last_iter)
                    print("iteration2", iteration2)

                    ## compute features of prototypes of the current task
                    map_Y_proto_cls = np.array([order_list.index(i) for i in Y_protoset_current])
                    current_eval_set = merge_images_labels(X_protoset_current, map_Y_proto_cls)
                    evalset.imgs = evalset.samples = current_eval_set
                    evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size, shuffle=False,
                                                             num_workers=args.num_workers, pin_memory=True)
                    num_samples = len(map_Y_proto_cls)
                    mapped_prototypes = compute_features(tg_feature_model, evalloader, num_samples, num_features)
                    D = mapped_prototypes #nbproto x 512
                    D_proto_cls = D / np.linalg.norm(D, axis=1, keepdims=True)

                    #D = mapped_prototypes.T # 512 x nbproto
                    #D = D / np.linalg.norm(D, axis=0)
                    #D_proto_cls = D.T # nbproto x 512

                    ## compute features of training data of the current task
                    idx = np.where(Y_train_ori == Y_protoset_current[0])
                    X_train_ori_cls = X_train_ori[idx]  # X_train_total[indices_train_10]
                    Y_train_ori_cls = Y_train_ori[idx]  # Y_train_total[indices_train_10]

                    map_Y_train_ori_cls = np.array([order_list.index(i) for i in Y_train_ori_cls])
                    current_eval_set = merge_images_labels(X_train_ori_cls, map_Y_train_ori_cls)
                    evalset.imgs = evalset.samples = current_eval_set
                    evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size, shuffle=False,
                                                             num_workers=args.num_workers, pin_memory=True)
                    num_samples = len(Y_train_ori_cls)
                    mapped_prototypes = compute_features(tg_feature_model, evalloader, num_samples, num_features)
                    D = mapped_prototypes
                    D_train_cls = D / np.linalg.norm(D, axis=1, keepdims=True)

                    #D = mapped_prototypes.T
                    #D = D / np.linalg.norm(D, axis=0)
                    #D_train_cls = D.T  # nbproto x 512
                    #mu = np.mean(D_train_cls, axis=0)  # feature mean of current training data

                    ### Calculate volume with DPP
                    L = np.dot(D_proto_cls, D_proto_cls.T)
                    det_L = np.linalg.det(L)
                    volume_cls = np.sqrt(det_L)
                    print("volume_cls", volume_cls)

                    if np.log(volume_cls) > -500:
                        volume_all_cls += np.log(volume_cls)
                        num_all_cls += 1


        #Average for all classes
        volume_all_cls = volume_all_cls / num_all_cls
        print("Averaged log Volume", round(volume_all_cls, 6))
