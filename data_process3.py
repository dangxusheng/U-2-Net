# -*- coding: utf-8 -*-
# @Time : 2023/8/20 14:25
# @Author : DangXS
# @Email : dangxusheng163@163.com
# @File : data_process.py
# @Project : mmsegmentation-deeptk

"""
根据大模型生成的分割结果，再进行标注调整，重新生成准确标签，进行训练

这个图片是 W*2 的图片，原图和标签都需要 切左一半保存

--001
    --1
        --001.src.show.jpg
        --001.src.show.jpg.json
        ...
"""


"""
python datasets/proejct_zhubao/data_process3.py
"""

import os, os.path as osp
import glob
import numpy as np
import cv2
from PIL import Image
import json
import shutil

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


def process():
    """
    解析标注的数据，并生成分割数据集

    # skip_names = ['似圆环状', '似圆球状', '似圆柱状' ]，这类要可视化看一下特殊处理
    """
    _path = 'E:/Projects/Dataset/SaliencyDetection/ZhuBao_TR/20230830/image'
    _seg_save_path = f'{_path}_label'

    NAMES = []
    COLORS = []
    SHAPES = []

    for _f in glob.iglob(f'{_path}/**/*.jpg', recursive=True):
        _img_path = _f
        fname = osp.basename(_f)

        _label_json = f'{_f}.json'
        assert osp.exists(_label_json)

        print(_img_path)

        fp = open(_label_json, 'r')
        json_data = json.load(fp)
        fp.close()

        if not json_data['valid']:
            print(f'{_img_path} is invalid, skip.')
            continue

        width, height = json_data['width'], json_data['height']

        step_1 = json_data.get('step_1', None)
        if step_1 is None:
            print(f'{_f} [step_1] is null, continue.')
            continue
        step_3 = json_data['step_1']
        assert step_3['toolName'] == 'polygonTool'
        step_3_polygons = []
        step_3_polygon_ids = []
        step_3_polygon_parent_ids = []
        for _res in step_3['result']:
            if not _res['valid']:
                print(f'polygon: {_res} is invalid, skip.')
                continue
            _polygon_points = []
            for _p in _res['pointList']:
                _polygon_points.append([_p['x'], _p['y']])
            step_3_polygons.append(_polygon_points)
            step_3_polygon_ids.append(_res['id'])
            step_3_polygon_parent_ids.append(_res['sourceID'])

        # show
        # src_img = cv2.imread(_img_path)
        src_img = Image.open(_img_path)
        src_img = np.asarray(src_img)
        src_img = src_img[:,:,::-1]
        src_img = np.ascontiguousarray(src_img)
        h,w = src_img.shape[:2]
        # TODO: 只取w//2 作为原图
        src_img = src_img[:,:w//2,:]

        import copy
        src_img2 = copy.deepcopy(src_img)
        assert src_img is not None

        shape_mask_img = np.zeros_like(src_img, dtype=np.uint8)
        # 第1层：琢型区域
        for _polygon_points in step_3_polygons:
            _polygon_points = np.asarray(_polygon_points,dtype=np.int32)
            cv2.fillPoly(shape_mask_img,[_polygon_points],(0,0,255))
            # cv2.addWeighted(src_img2,0.6,shape_mask_img,0.4,0, dst=src_img2)
            cv2.fillPoly(src_img2,[_polygon_points],(0,128,0))

        color_map = get_color_map_list(256)
        shape_mask_img = shape_mask_img[..., -1]  # [h,w]
        shape_mask_img[shape_mask_img > 0] = 1  # 就两类，前景和背景
        assert shape_mask_img.min() >= 0 and shape_mask_img.max() <= 255

        # src_img = np.concatenate([src_img2,src_img],axis=1)
        # cv2.imshow('src', src_img)
        # cv2.waitKey(0)

        # save img
        img_suffix = '.src.show.jpg'
        save_src_file = f'{_seg_save_path}/{_f[len(_path)+1:-len(img_suffix)]}.src.jpg'
        os.makedirs(osp.dirname(save_src_file), exist_ok=True)
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        src_img = Image.fromarray(src_img)
        src_img.save(save_src_file)

        json_data['width'] //= 2
        save_json_file = f'{_seg_save_path}/{_f[len(_path)+1:-len(img_suffix)]}.src.jpg.json'
        with open(save_json_file,'w',encoding='utf-8') as fp:
            json.dump(json_data, fp)



        save_seg_shape_label_file = f'{_seg_save_path}/{_f[len(_path)+1:-len(img_suffix)]}.label.png'
        os.makedirs(osp.dirname(save_seg_shape_label_file), exist_ok=True)
        shape_lbl_pil = Image.fromarray(shape_mask_img.astype(np.uint8), mode='P')
        shape_lbl_pil.putpalette(color_map)
        shape_lbl_pil.save(save_seg_shape_label_file)

    NAMES = set(NAMES)
    SHAPES = set(SHAPES)
    COLORS = set(COLORS)

    print(f'Names: {NAMES}')
    print(f'Shapes: {SHAPES}')
    print(f'Colors: {COLORS}')



if __name__ == '__main__':
    process()
