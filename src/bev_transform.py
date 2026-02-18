"""
bev_transform.py
code borrowed from the HC net paper & repo.
https://github.com/xlwangDev/HC-Net

"""
import torch
import numpy as np
import cv2
from matplotlib import pyplot as plt
from utils.torch_geometry import get_perspective_transform, warp_perspective, hard_code_perspective

def get_BEV_kitti_tensor(front_img, fov, pitch, scale, out_size, yaw_deg=None):
    Hp, Wp = front_img.shape[:2]
    Wo, Ho = int(Wp * scale), int(Wp * scale)

    frame = torch.from_numpy(front_img)
    fov = fov*torch.pi/180                               #
    theta = pitch*torch.pi/180             # Camera pitch angle

    f = Hp/2/torch.tan(torch.tensor(fov))
    phi = torch.pi/2 - fov
    delta = torch.pi/2+theta - torch.tensor(phi)
    l = torch.sqrt(f**2+(Hp/2)**2)
    h = l*torch.sin(delta)
    f_ = l*torch.cos(delta)

    ######################
    out = torch.zeros((2, 2, 2))

    y = (torch.ones((2, 2)).T * (torch.arange(0,Ho, step=Ho-1))).T
    x = torch.ones((2, 2)) * torch.arange(0, Wo, step=Wo-1)
    l0 = torch.ones((2, 2)) * Ho - y
    l1 = torch.ones((2, 2)) * f_+ l0

    f1_0 = torch.arctan(h/l1)
    f1_1 = torch.ones((2, 2)) * (torch.pi/2+theta) - f1_0
    y_ = l0*torch.sin(f1_0) / torch.sin(f1_1)
    j_p = torch.ones((2, 2)) * Hp - y_
    i_p = torch.ones((2, 2)) * Wp/2 -(f_+torch.sin(torch.tensor(theta))*(torch.ones((2, 2))*Hp-j_p))*(Wo/2*torch.ones((2, 2))-x)/l1

    out[:,:,0] = i_p.reshape((2, 2))
    out[:,:,1] = j_p.reshape((2, 2))

    four_point_org = out.permute(2,0,1)
    four_point_new = torch.stack((x,y), dim = -1).permute(2,0,1)
    four_point_org = four_point_org.unsqueeze(0).flatten(2).permute(0, 2, 1)
    four_point_new = four_point_new.unsqueeze(0).flatten(2).permute(0, 2, 1)
    H_original = get_perspective_transform(four_point_org, four_point_new)

    if yaw_deg is None:
        H_yaw = torch.eye(3).unsqueeze(0)
    else:
        H_yaw = hard_code_perspective(yaw_deg, out_size)

    scale1, scale2 = out_size/Wo,out_size/Ho
    T3 = np.array([[scale1, 0, 0], [0, scale2, 0], [0, 0, 1]]) # H1 from supplementary
    Homo = torch.matmul(torch.tensor(T3).unsqueeze(0).float(), H_original.float())
    # Homo = torch.matmul(Homo, H_original.float())
    BEV = warp_perspective(frame.permute(2,0,1).unsqueeze(0).float(), Homo, (out_size,out_size))
    return BEV[0]

def main():
    vert_fov_deg = 50
    theta_deg = 30.0  # pitch in bev
    scale = 4
    image_size=512
    device = torch.device("cpu")

    # take a random front view img from the SCAND dataset
    front_view_img = "/media/jim/Ironwolf/datasets/scand_data/images/A_Jackal_AHG_Library_Thu_Oct_28_1/img_1635452200038441393.png"
    front_frame =cv2.imread(front_view_img)
    img1 = get_BEV_kitti_tensor(front_frame, vert_fov_deg, theta_deg, scale, image_size).to(device).unsqueeze(0)
    bev_frame = img1[0].cpu().int().permute(1, 2, 0).numpy().astype(np.uint8)


    # ax00 = fig.add_subplot(gs[0, :2])
    # ax02 = fig.add_subplot(gs[0, 2])
    cv2.imshow(f"window", front_frame)
    cv2.waitKey()
    cv2.imshow(f"window", bev_frame)
    cv2.waitKey()


if __name__ == "__main__":
    main()