import itertools
import logging
import os.path as osp
import tempfile
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .builder import DATASETS
from .custom import CustomDataset

try:
    import pycocotools
    if not hasattr(pycocotools, '__sphinx_mock__'):  # for doc generation
        assert pycocotools.__version__ >= '12.0.2'
except AssertionError:
    raise AssertionError('Incompatible version of pycocotools is installed. '
                         'Run pip uninstall pycocotools first. Then run pip '
                         'install mmpycocotools to install open-mmlab forked '
                         'pycocotools.')

@DATASETS.register_module()
class RetinaVehicleDataset(CustomDataset):

    CLASSES = ('moto', 'truck', 'bike', 'car', 'pedestrian', 'bus', 'ba_gac', )
    def __init__(self, min_size=None, **kwargs):
        self.NK = 5
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        self.min_size = min_size
        self.gt_path = kwargs.get('gt_path')
        super(RetinaVehicleDataset, self).__init__(**kwargs)
        #print(self.cat2label)

    def _parse_ann_line(self, line, width, height):
        values = [float(x) for x in line.strip().split()[:-1]]
        label = line.strip().split()[-1]
        bbox = np.array(values[0:4], dtype=np.float32)
        kps = np.zeros( (self.NK,3), dtype=np.float32)
        ignore = False
        bbox = np.where(bbox>=0, bbox, 0)
        if bbox[2]>=width-1:
            bbox[2]=width-2
        if bbox[3]>=height-1:
            bbox[3]=height-2
        if bbox[2]-bbox[0]<=0 or bbox[3]-bbox[1]<0:
            ignore = True
        if self.min_size is not None:
            assert not self.test_mode
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w < self.min_size or h < self.min_size:
                ignore = True
                
        if len(values)>4:
            if len(values)>5:
                #print(values)
                kps = np.array( values[4:19], dtype=np.float32 ).reshape((self.NK,3))
                for li in range(kps.shape[0]):
                    if kps[li][0]>=width-1:
                        kps[li][0] = width-2
                    if kps[li][1]>=height-1:
                        kps[li][1] = height-2
                    if (kps[li,:]==-1).all():
                        #assert kps[li][2]==-1
                        kps[li][2] = 0.0 #weight = 0, ignore
                    else:
                        assert kps[li][2]>=0
                        kps[li][2] = 1.0 #weight
                        #if li==0:
                        #  landmark_num+=1
                        #if kps[li][2]==0.0:#visible
                        #  kps[li][2] = 1.0
                        #else:
                        #  kps[li][2] = 0.0
                    kps = np.where(kps>=0, kps, 0)
            else: #len(values)==5
                if not ignore:
                    ignore = (values[4]==1)
        else:
            assert self.test_mode

        assert bbox[bbox<0].shape[0]==0
        assert kps[kps<0].shape[0]==0
        return dict(bbox=bbox, kps=kps, ignore=ignore, cat=label)


    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """
        name = None
        bbox_map = {}
        for line in open(ann_file, 'r'):
            line = line.strip()
            if line.startswith('#'):
                value = line[1:].strip().split()
                name = value[0]
                width = int(value[1])
                height = int(value[2])

                bbox_map[name] = dict(width=width, height=height, objs=[])
                continue
            assert name is not None
            assert name in bbox_map
            bbox_map[name]['objs'].append(line)
        print('origin image size', len(bbox_map))
        data_infos = []
        for name in bbox_map:
            item = bbox_map[name]
            width = item['width']
            height = item['height']
            vals = item['objs']
            objs = []
            ignore = True
            for line in vals:
                data = self._parse_ann_line(line, width, height)
                if data is None:
                    continue
                objs.append( data ) #data is (bbox, kps, cat)
                if data['ignore']==False:
                    ignore = False
            if len(objs)==0 and not self.test_mode:
                continue
            if ignore:
                continue

            data_infos.append(dict(filename=name, width = width, height=height, objs = objs))
        return data_infos


    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        data_info = self.data_infos[idx]
        # print(data_info['filename'])

        bboxes = []
        keypointss = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in data_info['objs']:
            label = self.cat2label[obj['cat']]
            bbox = obj['bbox']
            keypoints = obj['kps']
            ignore = obj['ignore']
            if ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
                keypointss.append(keypoints)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
            keypointss = np.zeros((0, self.NK, 3))
        else:
            #bboxes = np.array(bboxes, ndmin=2) - 1
            bboxes = np.array(bboxes, ndmin=2)
            labels = np.array(labels)
            keypointss = np.array(keypointss, ndmin=3)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            #bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2)
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            keypointss = keypointss.astype(np.float32),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        # print(keypointss)
        return ann