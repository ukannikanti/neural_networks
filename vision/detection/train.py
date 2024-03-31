from anchor_utils import AnchorGenerator
from image_list import ImageList
from torchsummary import summary
import torchvision.datasets as datasets
import torch
from torch import nn, Tensor
import torchvision
from torchvision.models.resnet import ResNet50_Weights
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union
from rpn import RPNHead, RegionProposalNetwork
import _utils as det_utils
from collections import OrderedDict
from dataset import Dataset
from faster_rcnn import FasterRCNN
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
import torch.optim as optim

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    transforms.append(T.Resize((224, 224)))
    return T.Compose(transforms)

def get_backbone_model() -> torch.nn.Module:
    ## Define the model backbone to extract the features of a given input image. 
    resnet = torchvision.models.resnet50(weights = ResNet50_Weights.DEFAULT)
    modules = list(resnet.children())[:-2]
    backbone = nn.Sequential(*modules)
    backbone.out_channels = 2048
    return backbone

def get_dataset_loader() -> torch.utils.data.DataLoader:
    def collate_fn(batch):
        return tuple(zip(*batch))

    # Define a dataset loader
    dataset = Dataset("/Users/raghavakannikanti/opensource_2024/data/PennFudanPed", transforms=get_transform(train=True))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    return data_loader

def get_faster_rcnn_model() -> torch.nn.Module:
    # Define a anchor generator to generate at each spatial location in the feature maps, 
    # a set of anchor boxes (or anchors) is generated. In this case, 6(scales * aspect ratios) anchor boxes are generated.
    rpn_anchor_generator =  AnchorGenerator(sizes=((16, 32),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))

    # let's define what are the feature maps that we will use to perform the region of interest cropping,
    # as well as the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to be ['0']. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

    model = FasterRCNN(get_backbone_model(), num_classes=2, rpn_anchor_generator=rpn_anchor_generator, box_roi_pool=roi_pooler)
    return model

def train():
    """
    To understand the end-end training process, lets consider a single image for input & output calculations. 
        len(aspect_ratios) = 3 & len(scales) = 2 => 6 anchors are generated for every feature map location.
        # Feature extraction
            1. resnet_model(image) -> (batch, 2048, 25, 25) feature maps 
        # Region propasals
            rpn(feature maps) ->  {
                1. RPNHead(features) ->  objectness, pred_bbox_deltas. 
                        a. Every feature map location * number of anchors per location.  
                        b. Shape of the objectness (batch, 6, 25, 25)
                        c. Shape of the pred_bbox_deltas (batch, 6 * 4, 25, 25)
                2. anchor_generator -> [(6 * 25 * 25, 4)] anchors generated. list size == batch size. 
                3. Filter proposals
                        a. Ensure that the predicted bounding boxes are within the bounds of the image
                        b. Filter out redundant bounding boxes that overlap significantly and keep only the most confident ones.
                          (Non-Maximum Suppression)
                4. Bounding box encoding & decoding
                        For example, if the anchor box representation is [0.2, 0.5, 0.1, 0.2], 
                        and the representation of the ground-truth box corresponding to the anchor box is [0.25, 0.55, 0.08, 0.25],
                        the prediction target, which is the offset, should be [0.05, 0.05, -0.02, 0.05]. 
                        The object detection bounding box regressor is trying to learn how to predict this offset.
                        Decoding => Adds deltas to the anchor boxes to obtain the predicted bounding box
                        Encoding => Calculates the offsets or deltas between target bounding boxes and anchor boxes
                5. Use decoding to get the obtained predicted bounding boxes.
                6. Use encoding to get the deltas to compute the regression loss.
                7. for classification loss, prepare labels if the anchor boax match with GT box assign label 1 or else 0.
                
                return (predicted_boxes, loss) => (proposals, proposal_losses)
            }
        # Faster R-CNN Detector
            Filter propasals and select IOU >= threshold
            select training samples from propasals with +ve and -ve samples
            Region of Interest (ROI) pooling
                    Used for utilising single feature map for all the proposals generated by RPN in a single pass.
                    ROI pooling/Align solves the problem of fixed image size requirement for object detection network.
                    ROI pooling/Align produces the fixed-size feature maps from non-uniform inputs by doing max-pooling on the inputs.
                    Spatial information is preserved
                    Translation Invariance (invariant to small translations or shifts in the object's position within the ROI)
                    ROI Pooling/Align:
                        https://kaushikpatnaik.github.io/annotated/papers/2020/07/04/ROI-Pool-and-Align-Pytorch-Implementation.html
            standard classification + bounding box regression layers    
            }
    """
    model = get_faster_rcnn_model()
    model.train()
    images, targets = next(iter(get_dataset_loader()))
    images = list(image for image in images)
    targets = [{k: v for k, v in t.items()} for t in targets]
    output = model(images, targets)
    print(output)
    
train()