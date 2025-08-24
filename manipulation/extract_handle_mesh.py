# -*- coding: utf-8 -*-

import argparse
import random
import time
import scipy.misc as misc
import json
import os
import sys
import numpy as np
from subprocess import call
from collections import deque
import pickle as pkl
from manipulation.utils import load_obj

def render(data, cur_shape_dir):
    cur_part_dir = os.path.join(cur_shape_dir, 'textured_objs')
    cur_v_list = []; cur_f_list = []; cur_v_num = 0;
    if 'objs' in data.keys():
        for child in data['objs']:
            v, f = load_obj(os.path.join(cur_part_dir, child+'.obj'))
            # v -= center
            # v /= scale
            cur_v_list.append(v)
            cur_f_list.append(f+cur_v_num)
            cur_v_num += v.shape[0]
    elif 'children' in data.keys():
        for child in data['children']:
            v, f = render(child, cur_shape_dir)
            cur_v_list.append(v)
            cur_f_list.append(f+cur_v_num)
            cur_v_num += v.shape[0]
    else:
        return

    part_v = np.vstack(cur_v_list)
    part_f = np.vstack(cur_f_list)

    if not os.path.exists(os.path.join(cur_shape_dir, "parts_render")):
        os.makedirs(os.path.join(cur_shape_dir, "parts_render"))
    out_point_file = os.path.join(cur_shape_dir, "parts_render", str(data['id'])+str(data["name"])+'.obj')
    if not os.path.exists(out_point_file):
        with open(out_point_file, 'w') as fout:
            for i in range(part_v.shape[0]):
                fout.write('v %f %f %f\n' % (part_v[i, 0], part_v[i, 1], part_v[i, 2]))
            for i in range(part_f.shape[0]):
                fout.write('f %d %d %d\n' % (part_f[i, 0], part_f[i, 1], part_f[i, 2]))
                        
    return part_v, part_f

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', type=str, default='Microwave')
    parser.add_argument('--obj_id', type=str, default=None)
    args = parser.parse_args()
    
    if args.obj_id is not None:
        cur_shape_dir = "data/dataset/{}".format(args.obj_id)
        cur_result_json = os.path.join(cur_shape_dir, 'result.json')
        with open(cur_result_json, 'r') as fin:
            tree_hier = json.load(fin)[0]
        data = tree_hier
        render(data, cur_shape_dir)
    else:
        import json
        with open('data/partnet_mobility_dict.json', 'r') as fin:
            meta_dict = json.load(fin)
        all_objects = meta_dict[args.category]
        for obj_id in all_objects:
            cur_shape_dir = "data/dataset/{}".format(obj_id)
            print(obj_id)
            cur_result_json = os.path.join(cur_shape_dir, 'result.json')
            with open(cur_result_json, 'r') as fin:
                tree_hier = json.load(fin)[0]
            data = tree_hier
            render(data, cur_shape_dir)