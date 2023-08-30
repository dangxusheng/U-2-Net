import os
import os.path as osp
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms  # , utils

import numpy as np
from PIL import Image
import cv2
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET  # full size version 173.6 MB
from model import U2NETP  # small version u2net 4.7 MB

import shutil
import random
import json

"""

# deeptk 3090
cd /home/deeptk/projects/dxs/U-2-Net && conda activate py37_pytorch180
python infer_imgs.py

nohup python infer_imgs.py > run_infer_imgs.py.log 2>&1 &
PID: 28315

"""

FORE_THRES = 0.85

def get_color_map_list(num_classes):
    """
    Returns the color map for visualizing the segmentation mask,
    which can support arbitrary number of classes.
    Args:
        num_classes (int): Number of classes.
    Returns:
        (list). The color map.
    """

    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = color_map[3:]
    return color_map

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name, pred, d_dir, prefix):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    # get min bounding box
    mask_h, mask_w = np.where(predict_np >= FORE_THRES)
    x1 = np.min(mask_w)
    x2 = np.max(mask_w)
    y1 = np.min(mask_h)
    y2 = np.max(mask_h)
    mask_box = [x1, y1, x2, y2]

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    pb_np = np.array(imo)

    scale_hw = np.asarray(predict_np.shape) * 1.0 / np.asarray(image.shape[:2])
    scale_wh = list(scale_hw)[::-1]*2
    mask_box = np.asarray(mask_box) / np.asarray(scale_wh)
    mask_box = list(map(int, mask_box))

    if mask_box[2]-mask_box[0] < 1 or mask_box[3]-mask_box[1] < 1:
        print('invalid mask_box, skip.')
        return

    print(imo.size, mask_box)

    # show
    cv_img = np.array(imo, dtype=np.uint8)
    cv_img[:, :, 0][cv_img[:, :, 0] > 0] = 255
    cv_img[:, :, 1][cv_img[:, :, 1] > 0] = 255
    cv_img[:, :, 2][cv_img[:, :, 2] > 0] = 0
    mask = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    abs_filepath=image_name
    # img_suffix = '.src.jpg'
    img_suffix = '.jpg'

    # save show jpg
    cv_img = cv2.addWeighted(image, 0.8, cv_img, 0.2, 0)
    ret, thresh = cv2.threshold(mask, 0, 200, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cnt = contours[0]
    for cnt in contours:
        cv2.drawContours(cv_img, [cnt], 0, (0, 0, 255), 2)
    # cv2.rectangle(cv_img, tuple(mask_box[:2]), tuple(mask_box[2:]), (255, 0, 0), 2)
    out_src_jpg_file = f'{d_dir}/{abs_filepath[len(prefix)+1:-len(img_suffix)]}.src.show.jpg'
    print('out_src_jpg_file: ', out_src_jpg_file)
    os.makedirs(osp.dirname(out_src_jpg_file), exist_ok=True)
    # cv2.imwrite(out_src_jpg_file, cv_img)
    cv_img = np.concatenate([image,cv_img],axis=1)
    cv_img = cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB)
    _img = Image.fromarray(cv_img)
    _img.save(out_src_jpg_file)
    assert osp.exists(out_src_jpg_file)


    filename = osp.basename(image_name)
    _savefile = f'{d_dir}/{abs_filepath[len(prefix)+1:]}'
    print('_savefile: ', _savefile)

    color_map = get_color_map_list(256)
    lbl = mask
    lbl[lbl > 0] = 1  # 就两类，前景和背景

    out_png_file = f'{d_dir}/{abs_filepath[len(prefix)+1:-len(img_suffix)]}.label.png'
    os.makedirs(osp.dirname(out_png_file), exist_ok=True)
    out_src_png_file = f'{d_dir}/{abs_filepath[len(prefix)+1:-len(img_suffix)]}.src.jpg'
    shutil.copyfile(image_name, out_src_png_file)

    # Assume label ranges [0, 255] for uint8,
    if lbl.min() >= 0 and lbl.max() <= 255:
        lbl_pil = Image.fromarray(lbl.astype(np.uint8), mode='P')
        lbl_pil.putpalette(color_map)
        lbl_pil.save(out_png_file)
    else:
        raise ValueError(
            '[%s] Cannot save the pixel-wise class label as PNG. '
            'Please consider using the .npy format.' % out_png_file)

    # save crop img & seglabel & bbox
    src_w, src_h = image.shape[1], image.shape[0]
    _src_img = np.array(image, dtype=np.uint8)
    _src_img = cv2.cvtColor(_src_img, cv2.COLOR_RGB2BGR)
    const_expand = 1.25
    x1, y1, x2, y2 = mask_box
    bbox_h, bbox_w = y2 - y1, x2 - x1
    _size = max(bbox_h, bbox_w) * const_expand
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    _lx = max(0, cx - _size / 2.)
    _ly = max(0, cy - _size / 2.)
    _rx = min(src_w - 1, cx + _size / 2.)
    _ry = min(src_h - 1, cy + _size / 2.)

    x1, y1, x2, y2 = list(map(int, [_lx, _ly, _rx, _ry]))
    _src_crop_img = _src_img[y1:y2, x1:x2]
    save_src_jpg_file = f'{d_dir}/{abs_filepath[len(prefix)+1:-len(img_suffix)]}.src.crop.jpg'
    os.makedirs(osp.dirname(save_src_jpg_file), exist_ok=True)
    cv2.imwrite(save_src_jpg_file, _src_crop_img)

    lbl_crop = lbl[y1:y2, x1:x2]
    lbl_pil = Image.fromarray(lbl_crop, mode='P')
    lbl_pil.putpalette(color_map)
    save_label_png_file = f'{d_dir}/{abs_filepath[len(prefix)+1:-len(img_suffix)]}.label.crop.png'
    os.makedirs(osp.dirname(save_label_png_file), exist_ok=True)
    lbl_pil.save(save_label_png_file)

    bboxes_label_file = f'{d_dir}/{abs_filepath[len(prefix)+1:-len(img_suffix)]}.src.bbox.jpg.json'
    fp = open(bboxes_label_file, 'w', encoding='utf-8')
    bboxes = {
        'x1': mask_box[0],
        'y1': mask_box[1],
        'x2': mask_box[2],
        'y2': mask_box[3],
        'width': src_w,
        'height': src_h,
    }
    json.dump(bboxes, fp)
    fp.close()


def main():

    # --------- 1. get image path and name ---------
    model_name = 'u2net'  # u2netp

    image_dir = os.path.join(os.getcwd(), 'test_data', 'test_images')
    prediction_dir = os.path.join(
        os.getcwd(), 'test_data', model_name + '_results' + os.sep)
    model_dir = os.path.join(os.getcwd(), 'saved_models',
                             model_name, model_name + '.pth')

    img_name_list = glob.glob(image_dir + os.sep + '*')
    print(img_name_list)

    # --------- 2. dataloader ---------
    # 1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list,
                                        lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if (model_name == 'u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3, 1)
    elif (model_name == 'u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3, 1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:", img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        # normalization
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i_test], pred, prediction_dir)

        del d1, d2, d3, d4, d5, d6, d7


def test_imgs2():
    """
    使用分割大模型跑出来 mask和 bbox，生成标签，用于训练检测模型和分割模型
    """
    # --------- 1. model define ---------
    model_name = 'u2net'  # u2netp
    model_dir = './saved_models'
    assert osp.exists(model_dir)
    print("...load Net...")
    net = None
    if (model_name == 'u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3, 1)
        model_dir = f'{model_dir}/u2net.pth'
        # model_dir = f'{model_dir}/u2net_20230830/u2net_bce_itr_2000_train_2.385093_tar_0.356019.pth'
    elif (model_name == 'u2netp'):
        print("...load U2NEP---4.7 MB")
        model_dir = f'{model_dir}/u2netp.pth'
        net = U2NETP(3, 1)

    assert net is not None
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()
    transform = transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])

    # --------- 2. load images ---------
    root = 'E:/算法资源/训练/2020-08-28'
    img_dir = f'{root}/2'
    assert osp.exists(img_dir)
    N = 100
    img_suffix = '.src.jpg'
    prediction_dir = f'E:/算法资源/结果/2020-08-28/2-new'
    os.makedirs(prediction_dir, exist_ok=True)
    img_files = []
    # for _dname in [d.name for d in os.scandir(img_dir) if d.is_dir()]:
    #     # if _dname != '饰品名称': continue
    #     for _dname2 in [d.name for d in os.scandir(f'{img_dir}/{_dname}') if d.is_dir()]:
    #         # if _dname2 not in  ['手镯','手串','手链','项链']: continue
    #         _files = glob.glob(
    #             f'{img_dir}/{_dname}/{_dname2}/*{img_suffix}', recursive=True)
    #         random.shuffle(_files)
    #         if len(_files) > N:
    #             _files = random.sample(_files, N)
    #         img_files.extend(_files)

    img_suffix = '.jpg'
    img_files = glob.glob(f'{img_dir}/*{img_suffix}', recursive=True)
    assert len(img_files) > 0

    # --------- 3. inference for each image ---------
    for idx, f in enumerate(img_files):
        image = io.imread(f)
        # # 做滤波，把前景和背景对比度拉开一些
        # image_np = np.asarray(image, dtype=np.uint8)
        # # image_np = cv2.GaussianBlur(image_np,(5,5),1.5)
        # image_np = cv2.bilateralFilter(image_np, 5, 30, 5)
        # image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        label_3 = np.zeros(image.shape)
        # preprocess
        label = np.zeros(label_3.shape[0:2])
        if (3 == len(label_3.shape)):
            label = label_3[:, :, 0]
        elif (2 == len(label_3.shape)):
            label = label_3

        if (3 == len(image.shape) and 2 == len(label.shape)):
            label = label[:, :, np.newaxis]
        elif (2 == len(image.shape) and 2 == len(label.shape)):
            image = image[:, :, np.newaxis]
            label = label[:, :, np.newaxis]

        imidx = np.array([idx])
        sample = {'imidx': imidx, 'image': image, 'label': label}

        data_test = transform(sample)
        assert isinstance(data_test, dict)
        print(data_test['image'].shape)

        inputs_test = data_test['image']
        inputs_test = inputs_test[None, ...]
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = inputs_test.cuda()

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        # normalization
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)

        # save results to test_results folder
        save_output(f, pred, prediction_dir, prefix=img_dir)
        del d1, d2, d3, d4, d5, d6, d7

    print('done.')


def generate_label():
    """
    把标注的图片转化为 训练delabel，格式位黑白图mask
    """
    _image_path = 'G:/Project/DataSet/Saliency Detection/ZhuBao_TR/image'
    _label_path = 'G:/Project/DataSet/Saliency Detection/ZhuBao_TR/label'
    _image_suffix = '.src.jpg'
    _label_suffix = '.label.png'
    for f in glob.glob(f'{_image_path}/*{_image_suffix}',recursive=True):
        _img_path = f
        _lbl_path = f'{_label_path}/{osp.basename(f)[:-len(_image_suffix)]}{_label_suffix}'
        assert osp.exists(_lbl_path)
        print(_lbl_path)
        img = cv2.imread(_img_path)
        label = cv2.imread(_lbl_path, 0)
        print(np.max(label),np.min(label))
        if np.min(label)>1:
            label = np.asarray(label, dtype=np.float)/ np.max(label)
            label = np.asarray(label,dtype=np.uint8)
            label *=255
        print(np.max(label),np.min(label))
        print(label.shape)
        cv2.imshow('1',img)
        cv2.imshow('2',label)
        cv2.waitKey(0)


if __name__ == '__main__':
    test_imgs2()
    # generate_label()
