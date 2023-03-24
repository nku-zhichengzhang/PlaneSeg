import datetime
import math
import os
import random
import re

import numpy as np
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

import utils
from nms.nms_wrapper import nms
from roialign.roi_align.crop_and_resize import CropAndResizeFunction
import cv2
from models.modules import *
from utils import *

from models.model import *


class ECNN(nn.Module):
    def __init__(self, C1, C2, C3, C4, C5, out_channels, bilinear_upsampling=False):
        super(ECNN, self).__init__()
        self.out_channels = out_channels
        self.bilinear_upsampling = bilinear_upsampling
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.C4 = C4
        self.C5 = C5
        self.P6 = nn.MaxPool2d(kernel_size=1, stride=2)
        self.P5_conv1 = nn.Conv2d(2048, self.out_channels, kernel_size=1, stride=1)
        self.P5_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.P4_conv1 = nn.Conv2d(1024, self.out_channels, kernel_size=1, stride=1)
        self.P4_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.P3_conv1 = nn.Conv2d(512, self.out_channels, kernel_size=1, stride=1)
        self.P3_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.P2_conv1 = nn.Conv2d(256, self.out_channels, kernel_size=1, stride=1)
        self.P2_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )

        self.exconv_5 = nn.Conv2d(2049, self.out_channels, kernel_size=1, stride=1)
        self.exconv_4 = nn.Conv2d(1281, self.out_channels, kernel_size=1, stride=1)
        self.exconv_3 = nn.Conv2d(769, self.out_channels, kernel_size=1, stride=1)
        self.exconv_2 = nn.Conv2d(513, self.out_channels, kernel_size=1, stride=1)
        self.exconv_2_2 = nn.Conv2d(261, self.out_channels, kernel_size=1, stride=1)

        self.netScoreTwo = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreThr = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFou = torch.nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFiv = torch.nn.Conv2d(in_channels=2048, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.netCombine = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=4, out_channels=2, kernel_size=1, stride=1, padding=0),
            torch.nn.Sigmoid()
        )

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.down4 = nn.AvgPool2d(4)

        #print(C1) # 64
        #print(C2) # 256
        #print(C3) # 512
        #print(C4) # 1024
        #print(C5) # 2048

    def forward(self, x):
        c1 = self.C1(x)
        c2 = self.C2(c1)
        c3 = self.C3(c2)
        c4 = self.C4(c3)
        c5 = self.C5(c4)

        tenScoreTwo = self.netScoreTwo(c2)
        tenScoreThr = self.netScoreThr(c3)
        tenScoreFou = self.netScoreFou(c4)
        tenScoreFiv = self.netScoreFiv(c5)

        #tenScoreTwo_up = torch.nn.functional.interpolate(input=tenScoreTwo, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        tenScoreThr_up = torch.nn.functional.interpolate(input=tenScoreThr, size=(tenScoreTwo.shape[2], tenScoreTwo.shape[3]), mode='bilinear', align_corners=False)
        tenScoreFou_up = torch.nn.functional.interpolate(input=tenScoreFou, size=(tenScoreTwo.shape[2], tenScoreTwo.shape[3]), mode='bilinear', align_corners=False)
        tenScoreFiv_up = torch.nn.functional.interpolate(input=tenScoreFou, size=(tenScoreTwo.shape[2], tenScoreTwo.shape[3]), mode='bilinear', align_corners=False)

        edge = self.netCombine(torch.cat([ tenScoreTwo, tenScoreThr_up, tenScoreFou_up, tenScoreFiv_up ], 1))

        p5_out = self.exconv_5(torch.cat([c5, tenScoreFiv], 1))
        p4_out = self.exconv_4(torch.cat([c4, tenScoreFou, self.up2(p5_out)], 1))
        p3_out = self.exconv_3(torch.cat([c3, tenScoreThr, self.up2(p4_out)], 1))
        p2_out = self.exconv_2(torch.cat([c2, tenScoreTwo, self.up2(p3_out)], 1))

        #print(p2_out.shape, self.down4(x).shape, edge.shape)

        # Combine edge and rgb with p2_out
        p2_out = self.exconv_2_2(torch.cat([p2_out, self.down4(x), edge], 1))

        ## P6 is used for the 5th anchor scale in RPN. Generated by
        ## subsampling from P5 with stride of 2.
        p6_out = self.P6(p5_out)

        edge_up = self.up4(edge)

        #####
        return [p2_out, p3_out, p4_out, p5_out, p6_out, edge_up]
        #####



class MaskRCNN_ECNN(nn.Module):
    """Encapsulates the Mask RCNN model functionality.
    """

    def __init__(self, config, model_dir='test'):
        """
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        super(MaskRCNN_ECNN, self).__init__()
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.build(config=config)
        self.initialize_weights()
        self.loss_history = []
        self.val_loss_history = []

    def build(self, config):
        """Build Mask R-CNN architecture.
        """

        ## Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        ## Build the shared convolutional layers.
        ## Bottom-up Layers
        ## Returns a list of the last layers of each stage, 5 in total.
        ## Don't create the thead (stage 5), so we pick the 4th item in the list.
        resnet = ResNet("resnet101", stage5=True, numInputChannels=config.NUM_INPUT_CHANNELS)
        C1, C2, C3, C4, C5 = resnet.stages()

        ## Top-down Layers
        ## TODO: add assert to varify feature map sizes match what's in config
        self.fpn = ECNN(C1, C2, C3, C4, C5, out_channels=256, bilinear_upsampling=self.config.BILINEAR_UPSAMPLING)
        self.pn_transform2 = Bottleneck(256, 256 // 4)
        self.pn_transform3 = Bottleneck(256, 256 // 4)
        self.pn_transform4 = Bottleneck(256, 256 // 4)
        # self.fpn_base = FPN_base(C1, C2, C3, C4, C5, out_channels=256, bilinear_upsampling=self.config.BILINEAR_UPSAMPLING)
        ## Generate Anchors
        self.anchors = Variable(torch.from_numpy(utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                                                config.RPN_ANCHOR_RATIOS,
                                                                                config.BACKBONE_SHAPES,
                                                                                config.BACKBONE_STRIDES,
                                                                                config.RPN_ANCHOR_STRIDE)).float(),
                                requires_grad=False)
        if self.config.GPU_COUNT:
            self.anchors = self.anchors.cuda()

        ## RPN
        self.rpn = RPN(len(config.RPN_ANCHOR_RATIOS), config.RPN_ANCHOR_STRIDE, 256)

        ## Coordinate feature
        self.coordinates = nn.Conv2d(3, 64, kernel_size=1, stride=1)

        ## FPN Classifier
        self.debug = False
        self.classifier = Classifier(256, config.POOL_SIZE, config.IMAGE_SHAPE, config.NUM_CLASSES,
                                     config.NUM_PARAMETERS, debug=self.debug)

        ## FPN Mask
        self.mask = Mask(config, 256, config.MASK_POOL_SIZE, config.IMAGE_SHAPE, config.NUM_CLASSES)

        if self.config.PREDICT_DEPTH:
            if self.config.PREDICT_BOUNDARY:
                self.depth = Depth(num_output_channels=3)
            else:
                self.depth = Depth(num_output_channels=1)
                pass
            pass

            ## Fix batch norm layers

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False

        self.apply(set_bn_fix)

    def initialize_weights(self):
        """Initialize model weights.
        """

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def set_trainable(self, layer_regex, model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """

        for param in self.named_parameters():
            layer_name = param[0]
            trainable = bool(re.fullmatch(layer_regex, layer_name))
            if not trainable:
                param[1].requires_grad = False

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """

        ## Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        ## If we have a model path with date and epochs use them
        if model_path:
            ## Continue from we left of. Get epoch and date from the file name
            ## A sample model path might look like:
            ## /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/mask\_rcnn\_\w+(\d{4})\.pth"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6))

        ## Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        ## Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.pth".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{:04d}")

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            log_dir: The directory where events and weights are saved
            checkpoint_path: the path to the last checkpoint file
        """
        ## Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            return None, None
        ## Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        ## Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            return dir_name, None
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return dir_name, checkpoint

    def load_weights(self, filepath):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        if os.path.exists(filepath):
            state_dict = torch.load(filepath)
            try:
                self.load_state_dict(state_dict, strict=False)
            except:
                print('load only base model')
                try:
                    state_dict = {k: v for k, v in state_dict.items() if
                                  'classifier.linear_class' not in k and 'classifier.linear_bbox' not in k and 'mask.conv5' not in k}
                    state = self.state_dict()
                    state.update(state_dict)
                    self.load_state_dict(state)
                except:
                    print('change input dimension')
                    state_dict = {k: v for k, v in state_dict.items() if
                                  'classifier.linear_class' not in k and 'classifier.linear_bbox' not in k and 'mask.conv5' not in k and 'fpn.C1.0' not in k and 'classifier.conv1' not in k}
                    state = self.state_dict()
                    state.update(state_dict)
                    self.load_state_dict(state)
                    pass
                pass
        else:
            print("Weight file not found ...")
            exit(1)
        ## Update the log directory
        self.set_log_dir(filepath)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def detect(self, images, camera, mold_image=True, image_metas=None):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """

        ## Mold inputs to format expected by the neural network
        if mold_image:
            molded_images, image_metas, windows = mold_inputs(self.config, images)
        else:
            molded_images = images
            windows = [(0, 0, images.shape[1], images.shape[2]) for _ in range(len(images))]
            pass

        ## Convert images to torch tensor
        molded_images = torch.from_numpy(molded_images.transpose(0, 3, 1, 2)).float()

        ## To GPU
        if self.config.GPU_COUNT:
            molded_images = molded_images.cuda()

        ## Wrap in variable
        # molded_images = Variable(molded_images, volatile=True)

        ## Run object detection
        detections, mrcnn_mask, depth_np = self.predict([molded_images, image_metas, camera], mode='inference')

        if len(detections[0]) == 0:
            return [{'rois': [], 'class_ids': [], 'scores': [], 'masks': [], 'parameters': []}]

        ## Convert to numpy
        detections = detections.data.cpu().numpy()
        mrcnn_mask = mrcnn_mask.permute(0, 1, 3, 4, 2).data.cpu().numpy()

        ## Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks, final_parameters = \
                unmold_detections(self.config, detections[i], mrcnn_mask[i],
                                  image.shape, windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
                "parameters": final_parameters,
            })
        return results

    def predict(self, input, mode, use_nms=1, use_refinement=False, return_feature_map=False):
        molded_images = input[0]
        image_metas = input[1]

        if mode == 'inference':
            self.eval()
        elif 'training' in mode:
            self.train()

            ## Set batchnorm always in eval mode during training
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.apply(set_bn_eval)

        ## Feature extraction
        p2_out, p3_out, p4_out, p5_out, p6_out, edge = self.fpn(molded_images)
        ## Note that P6 is used in RPN, but not in the classifier heads.

        rpn_feature_maps = [p2_out, p3_out, p4_out, p5_out, p6_out]
        mrcnn_feature_maps = [p2_out, p3_out, p4_out, p5_out]

        feature_maps = [feature_map for index, feature_map in enumerate(rpn_feature_maps[::-1])]
        if self.config.PREDICT_DEPTH:
            depth_np = self.depth(feature_maps)
            if self.config.PREDICT_BOUNDARY:
                boundary = depth_np[:, 1:]
                depth_np = depth_np[:, 0]
            else:
                depth_np = depth_np.squeeze(1)
                pass
        else:
            depth_np = torch.ones((1, self.config.IMAGE_MAX_DIM, self.config.IMAGE_MAX_DIM)).cuda()
            pass

        ranges = self.config.getRanges(input[-1]).transpose(1, 2).transpose(0, 1)
        zeros = torch.zeros(3, (self.config.IMAGE_MAX_DIM - self.config.IMAGE_MIN_DIM) // 2,
                            self.config.IMAGE_MAX_DIM).cuda()
        ranges = torch.cat([zeros, ranges, zeros], dim=1)
        ranges = torch.nn.functional.interpolate(ranges.unsqueeze(0), size=(160, 160), mode='bilinear')
        ranges = self.coordinates(ranges * 10)

        ## Loop through pyramid layers
        layer_outputs = []  ## list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(self.rpn(p))

        ## Concatenate layer outputs
        ## Convert from list of lists of level outputs to list of lists
        ## of outputs across levels.
        ## e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        outputs = list(zip(*layer_outputs))
        outputs = [torch.cat(list(o), dim=1) for o in outputs]
        rpn_class_logits, rpn_class, rpn_bbox = outputs

        ## Generate proposals
        ## Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        ## and zero padded.
        proposal_count = self.config.POST_NMS_ROIS_TRAINING if 'training' in mode and use_refinement == False \
            else self.config.POST_NMS_ROIS_INFERENCE
        rpn_rois = proposal_layer([rpn_class, rpn_bbox],
                                  proposal_count=proposal_count,
                                  nms_threshold=self.config.RPN_NMS_THRESHOLD,
                                  anchors=self.anchors,
                                  config=self.config)

        if mode == 'inference':
            ## Network Heads
            ## Proposal classifier and BBox regressor heads
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_parameters = self.classifier(mrcnn_feature_maps,
                                                                                            rpn_rois, ranges)

            ## Detections
            ## output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in image coordinates
            detections = detection_layer(self.config, rpn_rois, mrcnn_class, mrcnn_bbox, mrcnn_parameters, image_metas)

            if len(detections) == 0:
                return [[]], [[]], depth_np
            ## Convert boxes to normalized coordinates
            ## TODO: let DetectionLayer return normalized coordinates to avoid
            ##       unnecessary conversions
            h, w = self.config.IMAGE_SHAPE[:2]
            scale = Variable(torch.from_numpy(np.array([h, w, h, w])).float(), requires_grad=False)
            if self.config.GPU_COUNT:
                scale = scale.cuda()
            detection_boxes = detections[:, :4] / scale

            ## Add back batch dimension
            detection_boxes = detection_boxes.unsqueeze(0)

            ## Create masks for detections
            mrcnn_mask, roi_features = self.mask(mrcnn_feature_maps, detection_boxes)

            ## Add back batch dimension
            detections = detections.unsqueeze(0)
            mrcnn_mask = mrcnn_mask.unsqueeze(0)
            return [detections, mrcnn_mask, depth_np]

        elif mode == 'training':

            gt_class_ids = input[2]
            gt_boxes = input[3]
            gt_masks = input[4]
            gt_parameters = input[5]

            ## Normalize coordinates
            h, w = self.config.IMAGE_SHAPE[:2]
            scale = Variable(torch.from_numpy(np.array([h, w, h, w])).float(), requires_grad=False)
            if self.config.GPU_COUNT:
                scale = scale.cuda()
            gt_boxes = gt_boxes / scale

            ## Generate detection targets
            ## Subsamples proposals and generates target outputs for training
            ## Note that proposal class IDs, gt_boxes, and gt_masks are zero
            ## padded. Equally, returned rois and targets are zero padded.
            rois, target_class_ids, target_deltas, target_mask, target_parameters = \
                detection_target_layer(rpn_rois, gt_class_ids, gt_boxes, gt_masks, gt_parameters, self.config)

            if len(rois) == 0:
                mrcnn_class_logits = Variable(torch.FloatTensor())
                mrcnn_class = Variable(torch.IntTensor())
                mrcnn_bbox = Variable(torch.FloatTensor())
                mrcnn_mask = Variable(torch.FloatTensor())
                mrcnn_parameters = Variable(torch.FloatTensor())
                if self.config.GPU_COUNT:
                    mrcnn_class_logits = mrcnn_class_logits.cuda()
                    mrcnn_class = mrcnn_class.cuda()
                    mrcnn_bbox = mrcnn_bbox.cuda()
                    mrcnn_mask = mrcnn_mask.cuda()
                    mrcnn_parameters = mrcnn_parameters.cuda()
            else:
                ## Network Heads
                ## Proposal classifier and BBox regressor heads
                # print([maps.shape for maps in mrcnn_feature_maps], target_parameters.shape)
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_parameters = self.classifier(mrcnn_feature_maps,
                                                                                                rois, ranges,
                                                                                                target_parameters)

                ## Create masks for detections
                mrcnn_mask, _ = self.mask(mrcnn_feature_maps, rois)

            return [rpn_class_logits, rpn_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox,
                    target_mask, mrcnn_mask, target_parameters, mrcnn_parameters, rois, depth_np]

        elif mode in ['training_detection', 'inference_detection']:
            gt_class_ids = input[2]
            gt_boxes = input[3]
            gt_masks = input[4]
            gt_parameters = input[5]

            ## Normalize coordinates
            h, w = self.config.IMAGE_SHAPE[:2]
            scale = Variable(torch.from_numpy(np.array([h, w, h, w])).float(), requires_grad=False)
            if self.config.GPU_COUNT:
                scale = scale.cuda()

            gt_boxes = gt_boxes / scale

            ## Generate detection targets
            ## Subsamples proposals and generates target outputs for training
            ## Note that proposal class IDs, gt_boxes, and gt_masks are zero
            ## padded. Equally, returned rois and targets are zero padded.

            rois, target_class_ids, target_deltas, target_mask, target_parameters = \
                detection_target_layer(rpn_rois, gt_class_ids, gt_boxes, gt_masks, gt_parameters, self.config)

            if len(rois) == 0:
                mrcnn_class_logits = Variable(torch.FloatTensor())
                mrcnn_class = Variable(torch.IntTensor())
                mrcnn_bbox = Variable(torch.FloatTensor())
                mrcnn_mask = Variable(torch.FloatTensor())
                mrcnn_parameters = Variable(torch.FloatTensor())
                if self.config.GPU_COUNT:
                    mrcnn_class_logits = mrcnn_class_logits.cuda()
                    mrcnn_class = mrcnn_class.cuda()
                    mrcnn_bbox = mrcnn_bbox.cuda()
                    mrcnn_mask = mrcnn_mask.cuda()
                    mrcnn_parameters = mrcnn_parameters.cuda()
            else:
                ## Network Heads
                ## Proposal classifier and BBox regressor heads
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_parameters, roi_features = self.classifier(
                    mrcnn_feature_maps, rois, ranges, pool_features=True)
                ## Create masks for detections
                mrcnn_mask, _ = self.mask(mrcnn_feature_maps, rois)
                pass

            h, w = self.config.IMAGE_SHAPE[:2]
            scale = Variable(torch.from_numpy(np.array([h, w, h, w])).float(), requires_grad=False)
            if self.config.GPU_COUNT:
                scale = scale.cuda()

            if use_refinement:
                mrcnn_class_logits_final, mrcnn_class_final, mrcnn_bbox_final, mrcnn_parameters_final, roi_features = self.classifier(
                    mrcnn_feature_maps, rpn_rois[0], ranges, pool_features=True)

                ## Add back batch dimension
                ## Create masks for detections
                detections, indices, _ = detection_layer(self.config, rpn_rois, mrcnn_class_final, mrcnn_bbox_final,
                                                         mrcnn_parameters_final, image_metas, return_indices=True,
                                                         use_nms=use_nms)
                if len(detections) > 0:
                    detection_boxes = detections[:, :4] / scale
                    detection_boxes = detection_boxes.unsqueeze(0)
                    detection_masks, _ = self.mask(mrcnn_feature_maps, detection_boxes)
                    roi_features = roi_features[indices]
                    pass
            else:
                mrcnn_class_logits_final, mrcnn_class_final, mrcnn_bbox_final, mrcnn_parameters_final = mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_parameters

                rpn_rois = rois
                detections, indices, _ = detection_layer(self.config, rpn_rois, mrcnn_class_final, mrcnn_bbox_final,
                                                         mrcnn_parameters_final, image_metas, return_indices=True,
                                                         use_nms=use_nms)

                if len(detections) > 0:
                    detection_boxes = detections[:, :4] / scale
                    detection_boxes = detection_boxes.unsqueeze(0)
                    detection_masks, _ = self.mask(mrcnn_feature_maps, detection_boxes)
                    roi_features = roi_features[indices]
                    pass
                pass

            valid = False
            if len(detections) > 0:
                positive_rois = detection_boxes.squeeze(0)

                gt_class_ids = gt_class_ids.squeeze(0)
                gt_boxes = gt_boxes.squeeze(0)
                gt_masks = gt_masks.squeeze(0)
                gt_parameters = gt_parameters.squeeze(0)

                ## Compute overlaps matrix [proposals, gt_boxes]
                overlaps = bbox_overlaps(positive_rois, gt_boxes)

                ## Determine postive and negative ROIs
                roi_iou_max = torch.max(overlaps, dim=1)[0]

                ## 1. Positive ROIs are those with >= 0.5 IoU with a GT box
                if 'inference' in mode:
                    positive_roi_bool = roi_iou_max > -1
                else:
                    positive_roi_bool = roi_iou_max > 0.2
                    pass
                detections = detections[positive_roi_bool]
                detection_masks = detection_masks[positive_roi_bool]
                roi_features = roi_features[positive_roi_bool]
                if len(detections) > 0:
                    positive_indices = torch.nonzero(positive_roi_bool)[:, 0]

                    positive_rois = positive_rois[positive_indices.data]

                    ## Assign positive ROIs to GT boxes.
                    positive_overlaps = overlaps[positive_indices.data, :]
                    roi_gt_box_assignment = torch.max(positive_overlaps, dim=1)[1]
                    roi_gt_boxes = gt_boxes[roi_gt_box_assignment.data, :]
                    roi_gt_class_ids = gt_class_ids[roi_gt_box_assignment.data]
                    roi_gt_parameters = gt_parameters[roi_gt_box_assignment.data]
                    roi_gt_parameters = self.config.applyAnchorsTensor(roi_gt_class_ids.long(), roi_gt_parameters)
                    ## Assign positive ROIs to GT masks
                    roi_gt_masks = gt_masks[roi_gt_box_assignment.data, :, :]

                    valid_mask = positive_overlaps.max(0)[1]
                    valid_mask = (valid_mask[roi_gt_box_assignment] == torch.arange(
                        len(roi_gt_box_assignment)).long().cuda()).long()
                    roi_indices = roi_gt_box_assignment * valid_mask + (-1) * (1 - valid_mask)

                    ## Compute mask targets
                    boxes = positive_rois
                    if self.config.USE_MINI_MASK:
                        ## Transform ROI corrdinates from normalized image space
                        ## to normalized mini-mask space.
                        y1, x1, y2, x2 = positive_rois.chunk(4, dim=1)
                        gt_y1, gt_x1, gt_y2, gt_x2 = roi_gt_boxes.chunk(4, dim=1)
                        gt_h = gt_y2 - gt_y1
                        gt_w = gt_x2 - gt_x1
                        y1 = (y1 - gt_y1) / gt_h
                        x1 = (x1 - gt_x1) / gt_w
                        y2 = (y2 - gt_y1) / gt_h
                        x2 = (x2 - gt_x1) / gt_w
                        boxes = torch.cat([y1, x1, y2, x2], dim=1)
                        pass
                    box_ids = Variable(torch.arange(roi_gt_masks.size()[0]), requires_grad=False).int()
                    if self.config.GPU_COUNT:
                        box_ids = box_ids.cuda()
                    roi_gt_masks = Variable(
                        CropAndResizeFunction(self.config.FINAL_MASK_SHAPE[0], self.config.FINAL_MASK_SHAPE[1], 0)(
                            roi_gt_masks.unsqueeze(1), boxes, box_ids).data, requires_grad=False)
                    roi_gt_masks = roi_gt_masks.squeeze(1)

                    roi_gt_masks = torch.round(roi_gt_masks)
                    valid = True
                    pass
                pass
            if not valid:
                detections = torch.FloatTensor()
                detection_masks = torch.FloatTensor()
                roi_gt_parameters = torch.FloatTensor()
                roi_gt_masks = torch.FloatTensor()
                roi_features = torch.FloatTensor()
                roi_indices = torch.LongTensor()
                if self.config.GPU_COUNT:
                    detections = detections.cuda()
                    detection_masks = detection_masks.cuda()
                    roi_gt_parameters = roi_gt_parameters.cuda()
                    roi_gt_masks = roi_gt_masks.cuda()
                    roi_features = roi_features.cuda()
                    roi_indices = roi_indices.cuda()
                    pass
                pass

            info = [rpn_class_logits, rpn_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox,
                    target_mask, mrcnn_mask, target_parameters, mrcnn_parameters, detections, detection_masks,
                    roi_gt_parameters, roi_gt_masks, rpn_rois, roi_features, roi_indices]
            if return_feature_map:
                feature_map = mrcnn_feature_maps
                info.append(feature_map)
                pass

            info.append(depth_np)
            if self.config.PREDICT_BOUNDARY:
                info.append(boundary)
                pass
            info.append(edge)
            return info
