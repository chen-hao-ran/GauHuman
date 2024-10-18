import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pickle

def check_bound_img():
    os.makedirs("output/bound_img_check", exist_ok=True)
    for i in range(96):
        if i < 10:
            img = "0000" + str(i) + ".png"
        elif i < 100:
            img = "000" + str(i) + ".png"
        image = cv2.imread(f'output/basketball28_Camera04/human1_96_pose_correction_lbs_offset_split_clone_merge_prune/test/ours_1200/renders/{img}')
        bound_mask = cv2.imread(f'output/basketball28_Camera04/human1_96_pose_correction_lbs_offset_split_clone_merge_prune/bound_mask_check/{i}.png')

        mask = bound_mask > 0
        masked_image = np.zeros_like(image)
        masked_image[mask] = image[mask]

        # SAVE
        cv2.imwrite(f'output/bound_img_check/{i}.png', masked_image)

    #     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #     plt.show()
    #     plt.pause(2)
    #     plt.clf()
    # plt.close

def check_smpl_rot(path):
    with open(path, 'rb') as file:
        smpl_rot = pickle.load(file)
    print(1)

if __name__ == '__main__':
    # check_bound_img()
    check_smpl_rot('output/basketball28_Camera04/human1_96_pose_correction_lbs_offset_split_clone_merge_prune/smpl_rot/iteration_1200/smpl_rot.pickle')