#!/opt/conda/envs/point-cloud/bin/python

# 本文件功能是把ModelNet中的.off文件转存成.ply文件
# 如果下载的不是.off文件，则补需要执行此文件

import os
import argparse
import numpy as np
from plyfile import PlyData
from plyfile import PlyElement

# 功能：从off文件中读取点云信息
# 输入：
#     filename:off文件名
def read_off(filename):
    points = []
    faces = []
    with open(filename, 'r') as f:
        first = f.readline()
        if (len(first) > 4): 
            n, m, c = first[3:].split(' ')[:]
        else:
            n, m, c = f.readline().rstrip().split(' ')[:]
        n = int(n)
        m = int(m)
        for i in range(n):
            value = f.readline().rstrip().split(' ')
            points.append([float(x) for x in value])
        for i in range(m):
            value = f.readline().rstrip().split(' ')
            faces.append([int(x) for x in value])
    points = np.array(points)
    faces = np.array(faces)
    return points, faces

# 功能：把点云信息写入ply文件，只写points
# 输入：
#     pc:点云信息
#     filename:文件名
def export_ply(pc, filename):
    vertex = np.zeros(pc.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    for i in range(pc.shape[0]):
        vertex[i] = (pc[i][0], pc[i][1], pc[i][2])
    ply_out = PlyData([PlyElement.describe(vertex, 'vertex', comments=['vertices'])])
    ply_filename = filename[:-4] + '.ply'
    ply_out.write(ply_filename)

# 功能：把ModelNet数据集文件从.off格式改成.ply格式，只包含points
# 输入：
#     ply_data_dir: ply文件的存放路径
#     off_data_dir: off文件的存放地址
def write_ply_points_only_from_off(ply_data_dir, off_data_dir):
    cat = os.listdir(off_data_dir)
    for i in range(len(cat)):
        print('writing ', i+1, '/', len(cat), ':', cat[i])

        filename = os.path.join(off_data_dir, cat[i])
        out = os.path.join(
            ply_data_dir, 
            f'{os.path.splitext(cat[i])[0]}.ply'
        )
        points, faces = read_off(filename)
        export_ply(points,out)

# 功能：把点云信息写入ply文件，包括points和faces
# 输入：
#     pc：points的信息
#     fc: faces的信息
#     filename:文件名
def export_ply_points_faces(pc,fc,filename):
    vertex = np.zeros(pc.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    face = np.zeros(fc.shape[0], dtype=[('vertex_indices', 'i4', (3,))])
    
    for i in range(pc.shape[0]):
        vertex[i] = (pc[i][0], pc[i][1], pc[i][2])
    
    for i in range(fc.shape[0]):
        face[i] = (fc[i,1], fc[i,2], fc[i,3])
    
    ply_out = PlyData([PlyElement.describe(vertex, 'vertex',comments=['vertices']),
                        PlyElement.describe(face, 'face', comments=['faces'])])

    ply_filename = filename[:-4] + '.ply'
    ply_out.write(ply_filename)

# 功能：把ModelNet数据集文件从.off格式改成.ply格式，包含points和faces
# 输入：
#     ply_data_dir: ply文件的存放路径
#     off_data_dir: off文件的存放地址
def write_ply_points_faces_from_off(ply_data_dir, off_data_dir):
    cat = os.listdir(off_data_dir)
    for i in range(len(cat)):
        if not os.path.exists(os.path.join(ply_data_dir,cat[i],'train')):
            os.makedirs(os.path.join(ply_data_dir,cat[i],'train'))
        if not os.path.exists(os.path.join(ply_data_dir,cat[i],'test')):
            os.makedirs(os.path.join(ply_data_dir,cat[i],'test'))
    for i in range(len(cat)):
        print('writing ', i+1, '/', len(cat), ':', cat[i])
        filenames = os.listdir(os.path.join(off_data_dir, cat[i],'train'))
        for j,x in enumerate(filenames):
            filename = os.path.join(off_data_dir, cat[i], 'train', x)
            out = os.path.join(ply_data_dir, cat[i], 'train', x)
            points, faces = read_off(filename)
            export_ply_points_faces(points,faces,out)
        filenames = os.listdir(os.path.join(off_data_dir, cat[i],'test'))
        for j,x in enumerate(filenames):
            filename = os.path.join(off_data_dir, cat[i], 'test', x)
            out = os.path.join(ply_data_dir, cat[i], 'test', x)
            points, faces = read_off(filename)
            export_ply_points_faces(points,faces,out)

def get_arguments():
    """ gets command line arguments.
    :return:
    """

    # init parser:
    parser = argparse.ArgumentParser("Downsample ModelNet40 by category.")

    # add required and optional groups:
    required = parser.add_argument_group('Required')
    optional = parser.add_argument_group('Optional')

    # add required:
    required.add_argument(
        "-i", dest="input", help="Input path of ModelNet 40 dataset in off format.",
        required=True
    )
    required.add_argument(
        "-o", dest="output", help="Output path of ModelNet 40 dataset in ply format.",
        required=True
    )

    # parse arguments:
    return parser.parse_args()


if __name__ == '__main__':
    # parse arguments:
    arguments = get_arguments()

    write_ply_points_only_from_off(arguments.output, arguments.input)