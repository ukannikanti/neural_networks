# from anchor_utils import AnchorGenerator
# from image_list import ImageList
# from torchsummary import summary
# import torchvision.datasets as datasets

# import torch
# from torch import nn, Tensor
# from torchvision import transforms, models
# from torchvision.models.resnet import ResNet50_Weights
# from PIL import Image
# from typing import Dict, List, Optional, Tuple, Union
# from rpn import RPNHead, RegionProposalNetwork
# import _utils as det_utils
# from collections import OrderedDict

# # read the image
# def read_image_to_tensor():
#     transform = transforms.Compose([
#          transforms.Resize((224, 224)),
#         transforms.ToTensor(),           
#     ])
#     image = Image.open("/Users/raghavakannikanti/opensource_2024/neural_networks_apps/vision/detection/image.jpeg")
#     tensor = transform(image)
#     return torch.unsqueeze(tensor, 0)

# def get_original_image_sizes(val):
#     original_image_sizes: List[Tuple[int, int]] = []
#     original_image_sizes.append((val[0], val[1]))
#     return original_image_sizes


# rpn_anchor_generator =  AnchorGenerator(sizes=((32, 64),),
#                                   aspect_ratios=((0.5, 1.0, 2.0),))


# backbone = models.resnet50(weights = ResNet50_Weights.DEFAULT)
# backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
# backbone.output_channels = 2048

# image = read_image_to_tensor()
# rpn_head = RPNHead(backbone.output_channels, rpn_anchor_generator.num_anchors_per_location()[0])
# features = backbone(image)
# print(features.shape)

# image_size = image.shape[-2:]
# image_size_list: List[Tuple[int, int]] = []
# image_size_list.append((image_size[0], image_size[1]))
# image_list = ImageList(image, image_size_list)


# rpn_pre_nms_top_n_train=200,
# rpn_pre_nms_top_n_test=100,
# rpn_post_nms_top_n_train=200,
# rpn_post_nms_top_n_test=100,
# rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
# rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)


# features = OrderedDict([("0", features)])
# targets = [
#     {
#         'boxes': torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]]),  # Format: [xmin, ymin, xmax, ymax]
#         'labels': torch.tensor([1, 2])  # Class labels for each bounding box
#     }
# ]

# rpn = RegionProposalNetwork(
#             rpn_anchor_generator,
#             rpn_head,
#             0.7,
#             0.3,
#             256,
#             0.5,
#             rpn_pre_nms_top_n,
#             rpn_post_nms_top_n,
#             0.7,
#             score_thresh=0.0,
#         )

# boxes, losses = rpn(image_list, features, targets)

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
# ])
# train_dataset = datasets.CocoDetection(root='~/coco/train', annFile='~/coco/annotations/train.json', transform=transform)
# image, target = train_dataset[0]
# print(image, target)

# # anchors = rpn_anchor_generator(image_list, features)
# # num_images = len(anchors)
# # num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
# # num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
# # print(num_anchors_per_level_shape_tensors)
# # print(num_anchors_per_level)

# # objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
# # print(objectness.shape)
# # print(pred_bbox_deltas[10])


# # box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
# # proposals = box_coder.decode(pred_bbox_deltas.detach(), anchors)
# # proposals = proposals.view(num_images, -1, 4)
# # print(proposals[0][10])

# # boxes, scores = filter_proposals(proposals, objectness, image, num_anchors_per_level)

import torch
g = torch.Generator()
g.manual_seed(2)
indices = torch.randperm(10, generator=g).tolist()
print(indices)
