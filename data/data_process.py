import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
from utils.smpl_torch_batch import SMPLModel

def data_reader(dataset):
    cam_path = os.path.join(dataset, 'cam_parms.npz')
    cam_parms = np.load(cam_path)
    extri = cam_parms['extrinsic']
    intri = cam_parms['intrinsic']

    posemap128_path = os.path.join(dataset, 'query_posemap_128_cano_smpl.npz')
    posemap128 = np.load(posemap128_path)
    posmap128 = posemap128['posmap128']

    model_path = os.path.join(dataset, 'smpl_parms.pth')
    model = torch.load(model_path)

    joint_mat = torch.load('data/basketball28_Camera04/train/smpl_cano_joint_mat.pth')

    print(cam_parms.files)

def get_smpl_vertices(dataset):
    os.makedirs('output/smpl_vertices/0/smpl_vertices', exist_ok=True)
    os.makedirs('output/smpl_vertices/1/smpl_vertices', exist_ok=True)

    smpl = SMPLModel(device=torch.device('cpu'), model_path='assets/SMPL_NEUTRAL.pkl')
    model_path = os.path.join(dataset, 'smpl_parms.pth')
    model = torch.load(model_path)
    pose = model['gt_pose']
    shape = model['gt_shape']
    trans = model['gt_trans']
    for frame in range(96):
        for human in range(2):
            p = pose[frame][human].reshape(1, -1)
            s = shape[frame][human].reshape(1, -1)
            t = trans[frame][human].reshape(1, -1)
            verts, joints = smpl(s, p, t)
            verts = verts.cpu().detach().numpy().reshape(-1, 3).astype(np.float32)
            np.save(f'output/smpl/{human}/smpl_vertices/{frame}.npy', verts)
            np.savetxt(f'output/smpl/{human}/smpl_vertices/txt/{frame}.txt', verts)
    print(model)

def get_smpl_params(dataset):
    os.makedirs('output/smpl_vertices/0/smpl_params', exist_ok=True)
    os.makedirs('output/smpl_vertices/1/smpl_params', exist_ok=True)

    model_path = os.path.join(dataset, 'smpl_parms.pth')
    model = torch.load(model_path)
    pose = model['gt_pose']
    shape = model['gt_shape']
    trans = model['gt_trans']
    for frame in range(96):
        for human in range(2):
            sp = {}
            sp['poses'] = pose[frame][human].numpy().reshape(1, -1)
            sp['shapes'] = shape[frame][human].numpy().reshape(1, -1)
            sp['Th'] = trans[frame][human].numpy().reshape(1, -1)
            sp['Rh'] = pose[frame][human][0:3].numpy().reshape(1, -1)
            # sp['poses'][0:3] = 0
            np.save(f'output/smpl/{human}/smpl_params/{frame}.npy', sp)
    print(model)

def npy2txt(input, output):
    input_file = np.load(input)
    np.savetxt(output, input_file, delimiter=',', fmt='%.2f')

def get_img_from_vertices(dataset, input):
    cam_path = os.path.join(dataset, 'cam_parms.npz')
    cam_parms = np.load(cam_path)
    extri = cam_parms['extrinsic']
    intri = cam_parms['intrinsic']
    K = intri
    R = extri[:3, :3]
    T = extri[:3, 3]
    T[2] = 0

    os.makedirs('output/project_check', exist_ok=True)
    # PROJECT
    for i in range(96):
        path = os.path.join(input, f'{i}.npy')
        points = np.load(path).astype(np.float32)
        # pcd = o3d.io.read_point_cloud(os.path.join(input, f"{i}.ply"))
        # points = np.asarray(pcd.points).astype(np.float32)
        projected = np.dot(points, R.T) + T
        projected = np.dot(projected, K.T)
        projected = projected[:, :2] / projected[:, 2:]
        projected = projected.astype(np.int32)

        # 创建一个空白图像
        h = 1280
        w = 940
        image = np.zeros((h, w, 3), dtype=np.uint8)

        # 绘制投影的点
        for point in projected:
            cv2.circle(image, tuple(point), 5, (0, 255, 0), -1)
        cv2.imwrite(f'output/project_check/{i}.png', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # 显示图像
        # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # plt.show()
    pcd = o3d.io.read_point_cloud("output/basketball28_Camera04/human1_96_pose_correction_lbs_offset_split_clone_merge_prune/points3d.ply")
    points = np.asarray(pcd.points).astype(np.float32)
    print(points)
    projected = np.dot(points, R.T) + T
    projected = np.dot(projected, K.T)
    projected = projected[:, :2] / projected[:, 2:]
    projected = projected.astype(np.int32)
    h = 1280
    w = 940
    image = np.zeros((h, w, 3), dtype=np.uint8)
    for point in projected:
        # print(point)
        cv2.circle(image, tuple(point), 5, (0, 255, 0), -1)
    cv2.imwrite('output/project_check/999.png', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def get_annots(dataset, output):
    # 创建字典
    annots = {}
    cams = {}

    # 将cam info 存入字典
    K = []
    D = []
    dt = np.zeros((1, 5))
    R = []
    T = []
    for i in range(1):
        cam_path = os.path.join(dataset, 'cam_parms.npz')
        cam_parms = np.load(cam_path)
        extri = cam_parms['extrinsic']
        intri = cam_parms['intrinsic']
        K.append(intri)
        R.append(extri[:3, :3])
        T.append((extri[:3, 3]).reshape(3, 1))
        D.append(dt)
    cams['K'] = K
    cams['R'] = R
    cams['T'] = T
    cams['D'] = D
    annots['cams'] = cams

    # 讲ims存入字典
    imgs = []
    for i in range(96):
        it = i + 5
        if it < 10:
            num = "00000" + str(it)
        elif it < 100:
            num = "0000" + str(it)
        elif it < 1000:
            num = "000" + str(it)
        else:
            num = "00" + str(it)
        img = []
        sub_ims = {}
        for j in range(1):
            path = os.path.join("images", "{}.jpg".format(num))
            img.append(path)
        sub_ims['ims'] = img
        imgs.append(sub_ims)
    annots['ims'] = imgs

    # SAVE
    np.save(output, annots)

    # CHECK
    data = np.load(output, allow_pickle=True).item()
    print("successfully saved annots.npy")

def ply2txt(dataset, input, output):
    model_path = os.path.join(dataset, 'smpl_parms.pth')
    model = torch.load(model_path)
    pose = model['gt_pose']
    Rh = pose[0, 1, :3].numpy().reshape(1, 3)
    R = cv2.Rodrigues(Rh)[0].astype(np.float32)

    pcd = o3d.io.read_point_cloud(input)
    points = np.asarray(pcd.points).astype(np.float32)
    points = np.dot(points, R.T)
    np.savetxt(output, points)

if __name__ == '__main__':
    dataset = 'data/basketball28_Camera04/train'
    # data_reader(dataset)
    get_smpl_params(dataset)
    get_smpl_vertices(dataset)
    # npy2txt('output/smpl/0/smpl_vertices/95.npy', 'output/95.txt')
    # get_img_from_vertices(dataset, 'output/smpl/1/smpl_vertices')
    # get_annots(dataset, os.path.join(dataset, 'annots.npy'))
    # ply2txt(dataset,'output/basketball28_Camera04/human1_96_pose_correction_lbs_offset_split_clone_merge_prune/points3d.ply', 'output/basketball28_Camera04/human1_96_pose_correction_lbs_offset_split_clone_merge_prune/points3d_rot.txt')
    # data = np.load('data/basketball28_Camera04/train/smpl_params/8.npy', allow_pickle=True).item()