# -*- coding: utf-8 -*-
# @Time : 2023/8/20 14:25
# @Author : DangXS
# @Email : dangxusheng163@163.com
# @File : data_process.py
# @Project : mmsegmentation-deeptk

"""
根据客户生成的json，生成 区域分割标签, 包括 琢型 和颜色
--001
    --1
        --001.jpg
        --001.jpg.json
        --002.jpg
        --002.jpg.json
"""


"""
python datasets/proejct_zhubao/data_process.py
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
    _path = '/home/deeptk1/datasets/zhubao_dataset/beijing_labels/0818'
    _seg_save_path = '/home/deeptk1/datasets/zhubao_dataset/beijing_labels/0818_seg_dataset2'
    # _path = 'H:/deeptk_dataset/珠宝数据/beijing_labels/0818'
    # _seg_save_path = 'H:/deeptk_dataset/珠宝数据/beijing_labels/0818_seg_dataset'

    NAMES = []
    COLORS = []
    SHAPES = []

    skip_names = ['似圆环状', ]
    for _f in glob.iglob(f'{_path}/**/*.jpg', recursive=True):
        _img_path = _f
        fname = osp.basename(_f)

        # for _s in skip_names:
        #     if _s in fname:
        #         print(f'{_s} 标签不能直接用，要特殊处理.')
        #         continue



        _name,_color,_shape,_ = fname.replace(' ','').split('__')
        NAMES.append(_name)
        COLORS.append(_color)
        SHAPES.append(_shape)

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

        assert step_1['toolName'] == 'rectTool'

        allinfo_bboxes = []

        step_1_bboxes = []
        step_1_bboxes_ids = []
        for _res in step_1['result']:
            if not _res['valid']:
                print(f'bbox: {_res} is invalid, skip.')
                continue
            x1, y1 = _res['x'], _res['y']
            w, h = _res['width'], _res['height']
            step_1_bboxes.append([x1, y1, w, h])
            step_1_bboxes_ids.append(_res['id'])

            _res['child_polygons'] = []
            allinfo_bboxes.append(_res)

        step_2 = json_data.get('step_2', None)
        if step_2 is None:
            print(f'{_f} [step_2] is null, continue.')
            continue
        step_2 = json_data['step_2']
        assert step_2['toolName'] == 'polygonTool'
        step_2_polygons = []
        step_2_polygon_ids = []
        step_2_polygon_parent_ids = []
        for _res in step_2['result']:
            if not _res['valid']:
                print(f'polygon: {_res} is invalid, skip.')
                continue
            _polygon_points = []
            for _p in _res['pointList']:
                _polygon_points.append([_p['x'], _p['y']])
            step_2_polygons.append(_polygon_points)
            step_2_polygon_ids.append(_res['id'])
            step_2_polygon_parent_ids.append(_res['sourceID'])

            _res['child_polygons'] = []
            for _box in allinfo_bboxes:
                if _box['id'] == _res['sourceID']:
                    _box['child_polygons'].append(_res)


        step_3 = json_data.get('step_3', None)
        if step_3 is None:
            print(f'{_f} [step_3] is null, continue.')
            continue
        step_3 = json_data['step_3']
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

            _res['child_polygons'] = []
            for _box in allinfo_bboxes:
                for _poly in _box['child_polygons']:
                    if _poly['id'] == _res['sourceID']:
                        _poly['child_polygons'].append(_res)

        # show
        # src_img = cv2.imread(_img_path)
        src_img = Image.open(_img_path)
        src_img = np.asarray(src_img)
        src_img = src_img[:,:,::-1]
        src_img = np.ascontiguousarray(src_img)
        import copy
        src_img2 = copy.deepcopy(src_img)
        assert src_img is not None

        has_many_box = len(allinfo_bboxes) > 1
        no_box = len(allinfo_bboxes) < 1
        # assert len(allinfo_bboxes) == 1, '只能有一个box.'
        if has_many_box | no_box:
            print(f'{_f} has_many_box=True or no_box=True')
            continue

        has_many_child_polygons1 = False
        has_many_child_polygons2 = False

        # 存在一些多个box的，只能取最大边界
        assert len(allinfo_bboxes) >0, '至少有一个box.'

        # 第一层：主体box
        _bboxes = []
        shape_mask_img = np.zeros_like(src_img, dtype=np.uint8)
        color_mask_img = np.zeros_like(src_img, dtype=np.uint8)
        for _box in allinfo_bboxes:
            # _box = allinfo_bboxes[0]
            x1, y1 = _box['x'], _box['y']
            x2, y2 = x1+_box['width'], y1+_box['height']
            x1,y1,x2,y2 = list(map(int, [x1,y1,x2,y2]))
            cv2.rectangle(src_img2,(x1,y1),(x2,y2),(0,0,255),2)
            _bboxes.append([x1,y1,x2,y2])

            # 第二层：琢型区域
            for _poly in _box['child_polygons']:
                _polygon_points = []
                for _p in _poly['pointList']:
                    _polygon_points.append([_p['x'], _p['y']])
                _polygon_points = np.asarray(_polygon_points,dtype=np.int32)
                cv2.fillPoly(shape_mask_img,[_polygon_points],(0,0,255))
                # cv2.addWeighted(src_img2,0.6,shape_mask_img,0.4,0, dst=src_img2)
                cv2.fillPoly(src_img2,[_polygon_points],(0,128,0))

                # 第三层：颜色区域
                for _poly2 in _poly['child_polygons']:
                    _polygon_points = []
                    for _p in _poly2['pointList']:
                        _polygon_points.append([_p['x'], _p['y']])
                    _polygon_points = np.asarray(_polygon_points, dtype=np.int32)
                    cv2.fillPoly(color_mask_img, [_polygon_points], (0, 0, 255))
                    # cv2.addWeighted(src_img2, 0.6, color_mask_img, 0.4, 0, dst=src_img2)
                    cv2.fillPoly(src_img2,[_polygon_points],(0,0,128))


        # src_img = np.concatenate([src_img2,src_img],axis=1)
        # cv2.imshow('src', src_img)
        # cv2.waitKey(0)

        # crop
        _bboxes = np.asarray(_bboxes)
        assert len(_bboxes)>0

        x1 = np.min(_bboxes[:,0])
        y1 = np.min(_bboxes[:,1])
        x2 = np.max(_bboxes[:,2])
        y2 = np.max(_bboxes[:,3])
        src_w, src_h = src_img.shape[1], src_img.shape[0]
        const_expand = 1.25
        bbox_h, bbox_w = y2 - y1, x2 - x1
        _size = max(bbox_h, bbox_w) * const_expand
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        _lx = max(0, cx - _size / 2.)
        _ly = max(0, cy - _size / 2.)
        _rx = min(src_w - 1, cx + _size / 2.)
        _ry = min(src_h - 1, cy + _size / 2.)

        x1, y1, x2, y2 = list(map(int, [_lx, _ly, _rx, _ry]))
        _src_crop_img = src_img[y1:y2, x1:x2]
        crop_src_img = _src_crop_img    #[h,w,c] bgr
        crop_shape_mask_img = shape_mask_img[y1:y2, x1:x2, -1]  #[h,w]
        crop_color_mask_img = color_mask_img[y1:y2, x1:x2, -1]  #[h,w]
        color_map = get_color_map_list(256)
        crop_shape_mask_img[crop_shape_mask_img>0]=1    # 就两类，前景和背景
        crop_color_mask_img[crop_color_mask_img>0]=1    # 就两类，前景和背景
        assert crop_shape_mask_img.min() >= 0 and crop_shape_mask_img.max() <= 255
        assert crop_color_mask_img.min() >= 0 and crop_color_mask_img.max() <= 255
        shape_lbl_pil = Image.fromarray(crop_shape_mask_img.astype(np.uint8), mode='P')
        shape_lbl_pil.putpalette(color_map)

        color_lbl_pil = Image.fromarray(crop_color_mask_img.astype(np.uint8), mode='P')
        color_lbl_pil.putpalette(color_map)

        # save img
        img_suffix = '.jpg'
        save_src_file = f'{_seg_save_path}/{_f[len(_path)+1:-len(img_suffix)]}.src.jpg'
        os.makedirs(osp.dirname(save_src_file), exist_ok=True)
        shutil.copyfile(_f, save_src_file)

        shutil.copy(_label_json,osp.dirname(save_src_file))

        # src_img2 = cv2.cvtColor(src_img2, cv2.COLOR_BGR2RGB)
        # src_img2 = Image.fromarray(src_img2)
        # src_img2.save(save_src_file)

        save_seg_src_file = f'{_seg_save_path}/{_f[len(_path)+1:-len(img_suffix)]}.src.crop.jpg'
        save_seg_shape_label_file = f'{_seg_save_path}/{_f[len(_path)+1:-len(img_suffix)]}.label.shape.png'
        save_seg_color_label_file = f'{_seg_save_path}/{_f[len(_path)+1:-len(img_suffix)]}.label.color.png'
        os.makedirs(osp.dirname(save_seg_src_file), exist_ok=True)
        # cv2.imwrite(save_seg_src_file, crop_src_img)  # 中文路径写入失败
        crop_src_img = cv2.cvtColor(crop_src_img,cv2.COLOR_BGR2RGB)
        crop_src_img = Image.fromarray(crop_src_img)
        crop_src_img.save(save_seg_src_file)

        os.makedirs(osp.dirname(save_seg_shape_label_file), exist_ok=True)
        shape_lbl_pil.save(save_seg_shape_label_file)
        os.makedirs(osp.dirname(save_seg_color_label_file), exist_ok=True)
        color_lbl_pil.save(save_seg_color_label_file)

    NAMES = set(NAMES)
    SHAPES = set(SHAPES)
    COLORS = set(COLORS)

    print(f'Names: {NAMES}')
    print(f'Shapes: {SHAPES}')
    print(f'Colors: {COLORS}')



if __name__ == '__main__':
    process()
