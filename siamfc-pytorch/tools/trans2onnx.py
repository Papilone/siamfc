import cv2
import torch

from siamfc import ops
from siamfc.backbones import AlexNetV1
from siamfc.heads import SiamFC
from siamfc.siamfc import Net


net_path = 'pretrained/siamfc_alexnet_e50.pth'
model = Net(
    backbone=AlexNetV1(),
    # backbone=resnet50([2,3,4]),
    head=SiamFC(0.001))
model.load_state_dict(torch.load(
    net_path, map_location=lambda storage, loc: storage))
# img = cv2.imread('miles.png')
dummy_input1 = torch.randn(1, 3, 256, 256, device='cpu')
dummy_input2 = torch.randn(1, 3, 256, 256, device='cpu')
dummy_input = (dummy_input1,dummy_input2)
torch.onnx._export(model, dummy_input, "SiamFC.onnx", verbose=True)