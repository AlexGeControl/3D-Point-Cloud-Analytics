import numpy as np
import math
import os
import struct
from scipy.spatial.transform import Rotation
import open3d as o3d


def get_P_from_Rt(R, t):
    P = np.identity(4)
    P[0:3, 0:3] = R
    P[0:3, 3] = t
    return P


def is_registration_successful(P_pred_np, P_gt_np):
    rte, rre = get_P_diff(P_pred_np, P_gt_np)
    return rte<2.0 and rre<5.0, rte, rre


def get_P_diff(P_pred_np, P_gt_np):
    P_diff = np.dot(np.linalg.inv(P_pred_np), P_gt_np)
    t_diff = np.linalg.norm(P_diff[0:3, 3])

    r_diff = P_diff[0:3, 0:3]
    R_diff = Rotation.from_matrix(r_diff)
    angles_diff = np.sum(np.abs(R_diff.as_euler('xyz', degrees=True)))

    return t_diff, angles_diff


def visualize_pc_pair(src_np, dst_np):
    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(np.transpose(src_np))
    pcd_src.paint_uniform_color([1, 0, 0])

    pcd_dst = o3d.geometry.PointCloud()
    pcd_dst.points = o3d.utility.Vector3dVector(np.transpose(dst_np))
    pcd_dst.paint_uniform_color([0, 1, 0])

    o3d.visualization.draw_geometries([pcd_src, pcd_dst])


def read_oxford_bin(bin_path):
    '''
    :param path:
    :return: [x,y,z,nx,ny,nz]: 6xN
    '''
    data_np = np.fromfile(bin_path, dtype=np.float32)
    return np.transpose(np.reshape(data_np, (int(data_np.shape[0]/6), 6)))


def read_reg_results(file_path, splitter=','):
    reg_gt_list = []
    with open(file_path, 'r') as f:
        line = f.readline()
        while line:
            items = line.split(splitter)
            items = [x.strip() for x in items]
            reg_gt_list.append(items)
            line = f.readline()
    return reg_gt_list


def reg_result_row_to_array(reg_result_row):
    idx1 = int(reg_result_row[0])
    idx2 = int(reg_result_row[1])
    t = np.asarray([float(x) for x in reg_result_row[2:5]])
    q_wxyz = [float(x) for x in reg_result_row[5:9]]
    q_xyzw = np.asarray(q_wxyz[1:] + q_wxyz[:1])

    rot = Rotation.from_quat(q_xyzw)

    return idx1, idx2, t, rot


def evaluate_rt(gt_file_path, predict_file_path):
    gt_reg = read_reg_results(gt_file_path)
    predict_reg = read_reg_results(predict_file_path)
    assert len(gt_reg) == len(predict_reg)

    counter_successful = 0
    rte_sum = 0
    rre_sum = 0
    for i in range(1, len(predict_reg)):
        gt_row = gt_reg[i]
        gt_idx1, gt_idx2, gt_t, gt_rot = reg_result_row_to_array(gt_row)
        gt_P = get_P_from_Rt(gt_rot.as_matrix(), gt_t)

        predict_row = predict_reg[i]
        predict_idx1, predict_idx2, predict_t, predict_rot = reg_result_row_to_array(predict_row)
        predict_P = get_P_from_Rt(predict_rot.as_matrix(), predict_t)

        assert gt_idx1 == predict_idx1
        assert gt_idx2 == predict_idx2

        is_reg_succ, rte, rre = is_registration_successful(predict_P, gt_P)
        if is_reg_succ:
            counter_successful += 1
            rte_sum += rte
            rre_sum += rre

            # for debug
            print(predict_row)

    reg_success_rate = counter_successful/len(gt_reg)
    avg_rte = rte_sum/counter_successful
    avg_rre = rre_sum/counter_successful
    print("Registration successful rate: %.2f, successful counter: %d, \n"
          "average Relative Translation Error (RTE): %.2f, average Relative Rotation Error (RRE): %.2f"
          % (reg_success_rate, counter_successful, avg_rte, avg_rre))
    return reg_success_rate, avg_rte, avg_rre


def main():
    # We do NOT provide the "groundtruth.txt". This is the evaluation script we will be using.
    # You are required to provide your registration results via 'reg_result.txt'
    # In our provided `reg_result.txt`, we provides 3 ground truth registration results as an example.
    # This script provides some functions to read and visualize the registration results.
    dataset_path = '/workspace/data/registration_dataset'
    ground_truth_reg_result_path = os.path.join(dataset_path, 'reg_result.csv')
    your_reg_result_path = os.path.join(dataset_path, 'reg_result_yaogefad.csv')

    # evaluate registration performance
    if os.path.exists(ground_truth_reg_result_path):
        evaluate_rt(ground_truth_reg_result_path, your_reg_result_path)

    # visualize registration result
    visualize_row_idx = 2
    reg_list = read_reg_results(os.path.join(dataset_path, 'reg_result.txt'), splitter=',')
    idx1, idx2, t, rot = reg_result_row_to_array(reg_list[visualize_row_idx])

    src_np = read_oxford_bin(os.path.join(dataset_path, 'point_clouds', '%d.bin' % idx1))[0:3, :]
    dst_np = read_oxford_bin(os.path.join(dataset_path, 'point_clouds', '%d.bin' % idx2))[0:3, :]
    dst_trans_np = np.dot(rot.as_matrix(), dst_np) + np.expand_dims(t, 1)

    visualize_pc_pair(src_np, dst_trans_np)


if __name__ == '__main__':
    main()