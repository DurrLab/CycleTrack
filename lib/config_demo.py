from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from dataset.dataset_factory import dataset_factory

class config(object):
    
    def __init__(self):
        self.data_dir='/cis/home/lhuang/my_documents/Cytometry/CenterTrack/data'
        self.dataset_version='train'
        self.tracking=True

        ############################
        # basic experiment setting #
        ############################

        self.task='tracking'
        #see lib/dataset/dataset_facotry for available datasets
        self.dataset='cell_tracking_demo'
        self.test_dataset='cell_tracking_demo'
        self.exp_id='cell_tracking_demo'
        self.test=False
        ##'level of visualization.'
        ##'1: only show the final detection results'
        ##'2: show the network output features'
        ##'3: use matplot to display' # useful when lunching training with ipython notebook
        ##'4: save all visualizations to disk'
        self.debug=4
        self.no_pause=False
        #'path to image/ image folders/ video. ''or "webcam"'
        self.demo=''
        #'path to pretrained model'
        #self.load_model='/cis/home/lhuang/my_documents/Cytometry/CenterTrack/exp/tracking/cell_tracking/logs_2020-06-22-02-38/model_232.pth'
        self.load_model='/cis/home/lhuang/my_documents/Cytometry/CenterTrack/exp/tracking/cell_tracking1/logs_2020-07-20-00-31/model_260.pth'
        #self.load_model='/cis/home/lhuang/my_documents/Cytometry/CenterTrack/exp/tracking/cell_tracking1/logs_2020-07-19-02-41/model_474.pth'
        #self.load_model='/cis/home/lhuang/my_documents/Cytometry/CenterTrack/models/pretrained.pth'
        #'resume an experiment. Reloaded the optimizer parameter and set load_model to model_last.pth in the exp dir if load_model is empty.'
        self.resume=False

        ##########
        # system #
        ##########

        #'-1 for CPU, use comma for multiple gpus'
        self.gpus='1' 
        #'dataloader threads. 0 for single-thread.'
        self.num_workers=4
        #'disable when the input size is not fixed.'
        self.not_cuda_benchmark=False
        #'random seed'from CornerNet
        self.seed=317
        #'used when training in slurm clusters.'
        self.not_set_cuda_env=False

        #######
        # log #
        #######
        #'disable progress bar and print to screen.'
        self.print_iter=1
        #'save model to disk every 5 epochs.'
        self.save_all=True
        #visualization threshold
        self.vis_thresh=0.3
        #choices=['white', 'black']
        self.debugger_theme='white' 
        self.eval_val=False
        self.save_imgs=''
        self.save_img_suffix=''
        self.skip_first=-1
        self.save_video=True
        self.save_framerate=30
        self.resize_video=False
        self.video_h=800
        self.video_w=1280
        self.transpose_video=False
        self.show_track_color=True
        self.not_show_bbox=True
        self.not_show_number=False
        self.qualitative=False
        self.tango_color=False

        #########
        # model #
        #########
        #model architecture. Currently tested res_18 | res_101 | resdcn_18 | resdcn_101 |dlav0_34 | dla_34 | hourglass
        self.arch='dla_34'
        self.dla_node='dcn' 
        #'conv layer channels for output head'
        #'0 for no conv layer'
        #-1 for default setting: '
        #'64 for resnets and 256 for dla.'
        self.head_conv=-1
        self.num_head_conv=1
        self.head_kernel=3
        # output stride. Currently only supports 4
        self.down_ratio=4
        self.not_idaup=False
        self.num_classes=-1
        self.num_layers=101
        self.backbone='dla34'
        self.neck='dlaup'
        self.msra_outchannel=256
        self.efficient_level=0
        self.prior_bias=-4.6 # -2.19

        ###########
        # dataset #
        ###########

        #'not use the random crop data augmentation from CornerNet.'
        self.not_rand_crop=True
        #'used when the training dataset has inbalanced aspect ratios.'
        self.not_max_crop=True
        #'when not using random crop, 0.1 apply shift augmentation.'
        self.shift=0.0
        #when not using random crop, 0.4 apply scale augmentation.
        self.scale=0.0
        #'probability of applying rotation augmentation.'
        self.aug_rot=0.1 
        #'when not using random crop apply rotation augmentation.'
        self.rotate=0.0
        #probability of applying flip augmentation.
        self.flip=0.0
        #'not use the color augmenation from CornerNet'
        self.no_color_aug=True
        
        #########
        # input #
        #########
        
        #'input height and width. -1 for default from dataset. Will be overriden by input_h | input_w'
        self.input_res=-1
        #'input height. -1 for default from dataset.'
        self.input_h=-1
        #'input width. -1 for default from dataset.'
        self.input_w=-1
        self.dataset_version=''
        
        #########
        # train #
        #########
        
        self.optim='adam'
        #learning rate for batch size 32.
        self.lr=1e-4
        #drop learning rate by 10.
        self.lr_step='40,120,200,280'
        #when to save the model to disk
        self.save_point='30,60,90,120,150,180,210,240,270,300'
        #total training epochs
        self.num_epochs=300
        self.batch_size=16
        #batch size on the master gpu.
        self.master_batch_size=-1
        #default: #samples / batch_size
        self.num_iters=-1
        #number of epochs to run validation
        self.val_intervals=10000
        #include validation in training and test on test set
        self.trainval=True
        self.ltrb=False
        self.ltrb_weight=0.1
        self.reset_hm=False
        self.reuse_hm=False
        self.use_kpt_center=False
        self.add_05=False
        self.dense_reg=1
        
        ########
        # loss #
        ########
        
        self.tracking_weight=1.0
        #regression loss: sl1 | l1 | l2
        self.reg_loss='l1'
        #loss weight for keypoint heatmaps
        self.hm_weight=2.0
        #loss weight for keypoint local offsets
        self.off_weight=1.0
        #loss weight for bounding box size
        self.wh_weight=0.4
        #loss weight for human pose offset
        self.hp_weight=1.0
        #loss weight for human keypoint heatmap
        self.hm_hp_weight=1.0
        self.amodel_offset_weight=1.0
        #loss weight for depth
        self.dep_weight=1.0
        #loss weight for 3d bounding box size
        self.dim_weight=1.0
        #loss weight for orientation
        self.rot_weight=1.0
        self.nuscenes_att=False
        self.nuscenes_att_weight=1.0
        self.velocity=False
        self.velocity_weight=1.0
        
        ############
        # Tracking #
        ############
        
        self.tracking=True
        self.pre_hm=True
        self.same_aug_pre=True
        self.zero_pre_hm=False
        self.hm_disturb=0.2
        self.lost_disturb=0.2
        self.fp_disturb=0.0
        self.pre_thresh=0.6
        self.track_thresh=0.35
        self.new_thresh=0.35
        self.max_frame_dist=-1
        self.ltrb_amodal=False
        self.ltrb_amodal_weight=0.1
        self.public_det=False
        self.no_pre_img=False
        self.zero_tracking=False
        self.hungarian=False
        self.max_age=-1
        self.tracking_base=False
        #search area
        self.track_size = 0.2
        self.item_size = 0.2
        self.cells_dist = 0.225
        
        ########
        # test #
        ########
      
        #flip data augmentation.
        self.flip_test=False
        #multi scale test augmentation.
        self.test_scales='1'
        #run nms in testing.
        self.nms=False
        #max number of output objects.
        self.K=100
        #not use parallal data pre-processing.
        self.not_prefetch_test=False
        self.fix_short=-1
        #keep the original resolution during validation.
        self.keep_res=False
        #if trained on nuscenes and eval on kitti
        self.map_argoverse_id=False
        self.out_thresh=-1.0
        self.depth_scale=1.0
        self.save_results=True
        self.load_results=''
        self.use_loaded_results=False
        self.ignore_loaded_cats=''
        #Used when convert to onnx
        self.model_output_list=False
        self.non_block_test=False
        self.vis_gt_bev=''
        #different validation split for kitti: 3dop | subcnn
        self.kitti_split='3dop'
        self.test_focal_length=-1   
        
    def parse(self):
        if self.test_dataset == '':
            self.test_dataset = self.dataset

        self.gpus_str = self.gpus
        self.gpus = [int(gpu) for gpu in self.gpus.split(',')]
        self.gpus = [i for i in range(len(self.gpus))] if self.gpus[0] >=0 else [-1]
        self.lr_step = [int(i) for i in self.lr_step.split(',')]
        self.save_point = [int(i) for i in self.save_point.split(',')]
        self.test_scales = [float(i) for i in self.test_scales.split(',')]
        self.save_imgs = [i for i in self.save_imgs.split(',')] \
          if self.save_imgs != '' else []
        self.ignore_loaded_cats = \
          [int(i) for i in self.ignore_loaded_cats.split(',')] \
          if self.ignore_loaded_cats != '' else []

        self.num_workers = max(self.num_workers, 2 * len(self.gpus))
        self.pre_img = False
        if 'tracking' in self.task:
            print('Running tracking')
            self.tracking = True
            self.out_thresh = max(self.track_thresh, self.out_thresh)
            self.pre_thresh = max(self.track_thresh, self.pre_thresh)
            self.new_thresh = max(self.track_thresh, self.new_thresh)
            self.pre_img = not self.no_pre_img
            print('Using tracking threshold for out threshold!', self.track_thresh)
            if 'ddd' in self.task:
                self.show_track_color = True

        self.fix_res = not self.keep_res
        print('Fix size testing.' if self.fix_res else 'Keep resolution testing.')

        if self.head_conv == -1: # init default head_conv
            self.head_conv = 256 if 'dla' in self.arch else 64

        self.pad = 127 if 'hourglass' in self.arch else 31
        self.num_stacks = 2 if self.arch == 'hourglass' else 1

        if self.master_batch_size == -1:
            self.master_batch_size = self.batch_size // len(self.gpus)
        rest_batch_size = (self.batch_size - self.master_batch_size)
        self.chunk_sizes = [self.master_batch_size]
        for i in range(len(self.gpus) - 1):
            slave_chunk_size = rest_batch_size // (len(self.gpus) - 1)
            if i < rest_batch_size % (len(self.gpus) - 1):
                slave_chunk_size += 1
            self.chunk_sizes.append(slave_chunk_size)
        print('training chunk_sizes:', self.chunk_sizes)

        if self.debug > 0:
            self.num_workers = 0
            self.batch_size = 1
            self.gpus = [self.gpus[0]]
            self.master_batch_size = -1

        # log dirs
        self.root_dir = '/cis/home/lhuang/my_documents/Cytometry/CenterTrack'
        self.data_dir = os.path.join(self.root_dir, 'data')
        self.exp_dir = os.path.join(self.root_dir, 'exp', self.task)
        self.save_dir = os.path.join(self.exp_dir, self.exp_id)
        self.debug_dir = os.path.join(self.save_dir, 'debug')

        if self.resume and self.load_model == '':
            self.load_model = os.path.join(self.save_dir, 'model_last.pth')
      
    def update_dataset_info_and_set_heads(self, dataset):
        self.num_classes = dataset.num_categories \
                          if self.num_classes < 0 else self.num_classes
        # input_h(w): opt.input_h overrides opt.input_res overrides dataset default
        input_h, input_w = dataset.default_resolution
        input_h = self.input_res if self.input_res > 0 else input_h
        input_w = self.input_res if self.input_res > 0 else input_w
        self.input_h = self.input_h if self.input_h > 0 else input_h
        self.input_w = self.input_w if self.input_w > 0 else input_w
        self.output_h = self.input_h // self.down_ratio
        self.output_w = self.input_w // self.down_ratio
        self.input_res = max(self.input_h, self.input_w)
        self.output_res = max(self.output_h, self.output_w)

        self.heads = {'hm': self.num_classes, 'reg': 2, 'wh': 2}

        if 'tracking' in self.task:
            self.heads.update({'tracking': 2})

        if 'ddd' in self.task:
            self.heads.update({'dep': 1, 'rot': 8, 'dim': 3, 'amodel_offset': 2})

        if 'multi_pose' in self.task:
            self.heads.update({
            'hps': dataset.num_joints * 2, 'hm_hp': dataset.num_joints,
            'hp_offset': 2})

        if self.ltrb:
            self.heads.update({'ltrb': 4})
        if self.ltrb_amodal:
            self.heads.update({'ltrb_amodal': 4})
        if self.nuscenes_att:
            self.heads.update({'nuscenes_att': 8})
        if self.velocity:
            self.heads.update({'velocity': 3})

        weight_dict = {'hm': self.hm_weight, 'wh': self.wh_weight,
                       'reg': self.off_weight, 'hps': self.hp_weight,
                       'hm_hp': self.hm_hp_weight, 'hp_offset': self.off_weight,
                       'dep': self.dep_weight, 'rot': self.rot_weight,
                       'dim': self.dim_weight,
                       'amodel_offset': self.amodel_offset_weight,
                       'ltrb': self.ltrb_weight,
                       'tracking': self.tracking_weight,
                       'ltrb_amodal': self.ltrb_amodal_weight,
                       'nuscenes_att': self.nuscenes_att_weight,
                       'velocity': self.velocity_weight}
        self.weights = {head: weight_dict[head] for head in self.heads}
        for head in self.weights:
            if self.weights[head] == 0:
                del self.heads[head]
        self.head_conv = {head: [self.head_conv \
          for i in range(self.num_head_conv if head != 'reg' else 1)] for head in self.heads}

        print('input h w:', self.input_h, self.input_w)
        print('heads', self.heads)
        print('weights', self.weights)
        print('head conv', self.head_conv)


    def init(self):
        # only used in demo
        train_dataset = 'cell_tracking1'
        self.parse()
        dataset = dataset_factory[train_dataset]
        self.update_dataset_info_and_set_heads(dataset)
