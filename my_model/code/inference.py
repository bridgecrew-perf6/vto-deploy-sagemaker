import os
import json

import torch
from torchvision import transforms

import numpy as np
from PIL import Image

from data_loader import RescaleT
from data_loader import ToTensorLab

from u2net import U2NET


def postprocess(d):
    """Normalize the predicted SOD probability map"""
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi) / (ma-mi)
    return dn


def model_fn(model_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = os.path.join(model_dir, 'my_model', 'model.pth')
    model = U2NET(3, 1)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)
    model.eval()
    return model


def input_fn(request_body, request_content_type):
    assert request_content_type == 'application/octet-stream', f'Request type: {request_content_type}'

    image = np.array(Image.open(request_body).convert('RGB'))
    transform = transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inputs = transform(image)
    inputs = inputs.unsqueeze(0)
    inputs = inputs.type(torch.FloatTensor)
    inputs = inputs.to(device)
    return {'inputs': inputs, 'width': image.shape[1], 'height': image.shape[0]}


def predict_fn(inputs_data, model):
    width = inputs_data['width']
    height = inputs_data['height']
    with torch.no_grad():
        d1, *_ = model(inputs_data['inputs'])

    # normalization
    pred = d1[:,0,:,:]
    pred = postprocess(pred)
    pred = pred.squeeze()
    predict_np = pred.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    imo = np.array(im.resize((width, height), resample=Image.BILINEAR))
    return imo


def output_fn(prediction, content_type):
    assert content_type == 'application/json'
    imo_list = prediction.tolist()
    return json.dumps(imo_list)
