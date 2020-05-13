import os
from pycocotools.coco import COCO
import numpy



def grab_ids(coco,use_ids):
    out = []
    for el in use_ids:
        for sacc in coco.imgToAnns[el]:
            out.append(sacc['id'])
    return out