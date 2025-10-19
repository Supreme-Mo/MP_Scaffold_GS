import numpy as np
import copy
import torch
import cv2
from utils.loss_utils import ssim
from utils.sh_utils import RGB2SH
import math
import torch.nn.functional as F


def world_to_view_screen(pts3D, K, RT_cam2):
    # print("pts3D",pts3D.max(), pts3D.min())
    wrld_X = RT_cam2.bmm(pts3D)
    xy_proj = K.bmm(wrld_X)
    # print("xy_proj",xy_proj.min(),xy_proj.max())

    # print("xy_proj",xy_proj.shape)

    # And finally we project to get the final result
    mask = (xy_proj[:, 2:3, :].abs() < 1E-2).detach()
    mask = mask.to(pts3D.device)
    mask.requires_grad_(False)

    zs = xy_proj[:, 2:3, :]

    mask_unsq = mask.unsqueeze(0).unsqueeze(0)

    if True in mask_unsq:
        zs[mask] = 1E-2
    sampler = torch.cat((xy_proj[:, 0:2, :] / zs, xy_proj[:, 2:3, :]), 1)

    # Remove invalid zs that cause nans
    if True in mask_unsq:
        sampler[mask.repeat(1, 3, 1)] = -10
    return sampler

def my_view_to_world_coord(projected_coors, K_inv, RTinv_cam1):
    # PERFORM PROJECTION
    # Project the world points into the new view
    # Transform into world coordinates
    cam1_X = K_inv.bmm(projected_coors)
    wrld_X = RTinv_cam1.bmm(cam1_X)




    return wrld_X


def MutiPlane_anchor_init(monodepth, xyz, view_camera, plane_num=24, sample_size=20,
                          muti_mode="neighbor", itera_num=1, feat_dim=32):
    """
    基于多平面的 Anchor 初始化
    输入: 
        monodepth   - 单目深度图 (H, W)
        xyz         - 原始点云 (N, 3)
        view_camera - 相机参数 (包含R, T, focal, original_image)
        plane_num   - 多平面划分数
        sample_size - 采样间隔
        muti_mode   - 深度采样模式 ["max", "random", "neighbor"]
        itera_num   - 下采样迭代次数
        feat_dim    - Anchor 特征维度 (默认3, RGB)
    输出:
        anchor_points: (M, 3) Tensor, Anchor 坐标
        anchor_feats : (M, feat_dim) Tensor, Anchor 特征 (默认RGB → SH)
    """

    # 原函数里生成的 Init_points 和 SHfeature 保留
    depth_min = monodepth.min()
    depth_max = monodepth.max()
    normal_depth = (depth_max - depth_min) / plane_num
    image = view_camera.original_image
    H, W = image.shape[1], image.shape[2]

    # 生成多平面 mask
    plane_mask = torch.zeros([H, W], device="cuda")
    for i in range(plane_num):
        plane_mask[monodepth <= ((i + 1) * normal_depth + depth_min)] += 1
    plane_mask = plane_mask - 1

    # 相机矩阵
    temp_R = copy.deepcopy(view_camera.R)
    temp_T = copy.deepcopy(view_camera.T)
    temp_R = np.transpose(temp_R)
    R = np.eye(4)
    R[:3, :3] = temp_R
    R[:3, 3] = temp_T
    H, W = view_camera.original_image.shape[1], view_camera.original_image.shape[2]
    K = torch.eye(4)
    K[0, 2] = W / 2
    K[1, 2] = H / 2
    K[0, 0] = view_camera.focal[1]
    K[1, 1] = view_camera.focal[0]
    K = torch.FloatTensor(K).unsqueeze(0).cuda()
    R = torch.FloatTensor(R).unsqueeze(0).cuda()

    # 点投影
    src_xyz_t = xyz.unsqueeze(0).permute(0, 2, 1)
    tempdata = torch.ones((src_xyz_t.shape[0], 1, src_xyz_t.shape[2])).cuda()
    src_xyz = torch.cat((src_xyz_t, tempdata), dim=1)
    xyz_sampler = world_to_view_screen(src_xyz, RT_cam2=R, K=K)
    sampler_temp = xyz_sampler[0, 0:2].transpose(1, 0).to(torch.int32)
    sampler_depth_temp = xyz_sampler[0, 2]

    # 屏幕边界过滤
    sampler_mask = torch.ones(sampler_temp.shape[0], device="cuda")
    sampler_mask[(sampler_temp[:, 1] >= H) | (sampler_temp[:, 0] >= W)] = 0
    sampler_mask[(sampler_temp[:, 1] < 0) | (sampler_temp[:, 0] < 0)] = 0
    thord_depth = sorted(list(sampler_depth_temp))[int(sampler_mask.shape[0] * 0.85)]
    sampler_mask[sampler_depth_temp > thord_depth] = 0
    sampler = sampler_temp[sampler_mask != 0]
    sampler_depth = sampler_depth_temp[sampler_mask != 0]

    # 多平面深度范围
    muti_plane_depth_range = torch.zeros(plane_num, 2).cuda()
    sampler_xy = sampler.clone()
    sampler_xy_label = np.zeros(sampler.shape[0])

    for i in range(sampler.shape[0]):
        plane_layer = int(plane_mask[sampler[i, 1], sampler[i, 0]])
        dep = sampler_depth[i]
        sampler_xy_label[i] = plane_layer
        if muti_plane_depth_range[plane_layer, 0] == 0 or dep < muti_plane_depth_range[plane_layer, 0]:
            muti_plane_depth_range[plane_layer, 0] = dep
        if muti_plane_depth_range[plane_layer, 1] == 0 or dep > muti_plane_depth_range[plane_layer, 1]:
            muti_plane_depth_range[plane_layer, 1] = dep

    # 坐标网格采样
    image1 = torch.zeros((H, W), device="cuda")
    image2 = torch.zeros((H, W), device="cuda")
    x_linspace1 = np.linspace(itera_num * int(sample_size / 2), H - 1,
                              int(H / sample_size) - 1 - itera_num).astype(int)
    y_linspace1 = np.linspace(itera_num * int(sample_size / 2), W - 1,
                              int(W / sample_size) - 1 - itera_num).astype(int)
    image1[x_linspace1, :] = 1
    image2[:, y_linspace1] = 1
    sampler_image = image1 + image2
    x_linspace = torch.linspace(0, W - 1, W).view(1, W).expand(H, W)
    y_linspace = torch.linspace(0, H - 1, H).view(H, 1).expand(H, W)
    image_axis = torch.stack([x_linspace, y_linspace], dim=2).to(torch.int32).cuda()
    sampler_axis = image_axis[sampler_image == 2]
    sampler_label = plane_mask[sampler_image == 2]

    sampler_depth_t = torch.zeros((sampler_label.shape[0], 1), device="cuda")
    RGB_feature = torch.zeros(sampler_axis.shape[0], 3).cuda()
    K_num = 5

    # 采样深度 & 提取特征
    for i in range(sampler_axis.shape[0]):
        layer_label = int(sampler_label[i])
        RGB_feature[i, :] = image[:, sampler_axis[i, 1], sampler_axis[i, 0]]
        if muti_mode == "random":
            sampler_depth_t[i] = muti_plane_depth_range[layer_label, 0] + \
                                 (muti_plane_depth_range[layer_label, 1] - muti_plane_depth_range[layer_label, 0]) * np.random.random()
        elif muti_mode == "max":
            sampler_depth_t[i] = muti_plane_depth_range[layer_label, 1]
        elif muti_mode == "neighbor":
            layer_sample_xy = sampler_xy[sampler_xy_label == layer_label]
            layer_depth = sampler_depth[sampler_xy_label == layer_label]
            mh_dis = layer_sample_xy - sampler_axis[i]
            if mh_dis.shape[0] <= K_num:
                mindep = muti_plane_depth_range[layer_label, 1]
            else:
                os_dis = torch.sqrt(mh_dis[:, 0] ** 2 + mh_dis[:, 1] ** 2)
                indices = torch.tensor(np.argpartition(os_dis.detach().cpu().numpy(), K_num)[:K_num])
                sorted_indices = indices[torch.argsort(os_dis[indices])]
                mindep = layer_depth[sorted_indices].max()
            sampler_depth_t[i] = mindep

    # 转换为世界坐标
    K_invs = torch.inverse(K)
    R_invs = torch.inverse(R)
    pts3D = torch.ones((1, 4, sampler_label.shape[0]), device="cuda")
    pts3D[:, 2:3, :] = sampler_depth_t.unsqueeze(0).permute(0, 2, 1)
    pts3D[:, 0:2, :] = (sampler_axis * sampler_depth_t).unsqueeze(0).permute(0, 2, 1)
    points = my_view_to_world_coord(pts3D, K_invs, R_invs)

    # === 输出 Anchor ===
    anchor_points = points[0, :3].permute(1, 0)             # (M, 3)
    #anchor_feats = RGB2SH(RGB_feature)                      # (M, feat_dim)，这里默认是 RGB→SH
    anchor_feats = torch.randn((anchor_points.shape[0], feat_dim), device="cuda") * 0.1
    return anchor_points, anchor_feats

# def MutiPlane_init(monodepth, xyz,view_camera,plane_num = 16, sample_size = 10,muti_mode = "max",itera_num= 16):
#     depth_min = monodepth.min()
#     depth_max = monodepth.max()
#     normal_depth = (depth_max-depth_min)/plane_num
#     image = view_camera.original_image
#     H, W = image.shape[1], image.shape[2]

#     plane_mask = torch.zeros([H,W],device="cuda")
#     for i in range(plane_num):
#         plane_mask[monodepth<=((i+1)*normal_depth+depth_min)] += 1
#     cv2.imwrite("plane_mask.png",plane_mask.detach().cpu().numpy()*16)
#     # exit()
#     plane_mask = plane_mask-1

#     temp_R = copy.deepcopy(view_camera.R)
#     temp_T = copy.deepcopy(view_camera.T)

#     temp_R = np.transpose(temp_R)
#     R = np.eye(4)
#     R[:3, :3] = temp_R
#     R[:3, 3] = temp_T
#     H, W = view_camera.original_image.shape[1], view_camera.original_image.shape[2]
#     K = torch.eye(4)

#     K[0, 2] = W / 2
#     K[1, 2] = H / 2
#     K[0, 0] = view_camera.focal[1]
#     K[1, 1] = view_camera.focal[0]

#     K = torch.FloatTensor(K).unsqueeze(0).cuda()
#     R = torch.FloatTensor(R).unsqueeze(0).cuda()

#     src_xyz_t = xyz.unsqueeze(0).permute(0, 2, 1)
#     tempdata = torch.ones((src_xyz_t.shape[0], 1, src_xyz_t.shape[2])).cuda()
#     src_xyz = torch.cat((src_xyz_t, tempdata), dim=1)

#     xyz_sampler = world_to_view_screen(src_xyz, RT_cam2=R, K=K)
#     sampler_temp = xyz_sampler[0, 0:2].transpose(1, 0)
#     sampler_temp = copy.deepcopy(torch.tensor(sampler_temp, dtype=int, device="cuda"))

#     sampler_depth_temp = xyz_sampler[0,2]

#     sampler_mask = torch.ones(sampler_temp.shape[0],device="cuda")

#     sampler_mask[sampler_temp[:, 1] >= H] = 0
#     sampler_mask[sampler_temp[:, 0] >= W] = 0
#     sampler_mask[sampler_temp[:, 1] < 0] = 0
#     sampler_mask[sampler_temp[:, 0] < 0] = 0
#     thord_depth = sorted(list(sampler_depth_temp))[int(sampler_mask.shape[0]*0.85)]

#     sampler_mask[sampler_depth_temp>thord_depth] = 0

#     sampler = sampler_temp[sampler_mask!=0]
#     sampler_depth = sampler_depth_temp[sampler_mask!=0]

#     muti_plane_depth_range = torch.zeros(plane_num,2).cuda()


#     sampler_xy = copy.deepcopy(sampler)
#     sampler_xy_label = np.zeros(sampler.shape[0])

#     for i in range(sampler.shape[0]):
#         plane_layer = int(plane_mask[sampler[i,1],sampler[i,0]])
#         dep = sampler_depth[i]

#         sampler_xy_label[i] = plane_layer

#         if muti_plane_depth_range[plane_layer,0] == 0 or dep<muti_plane_depth_range[plane_layer,0]:
#             muti_plane_depth_range[plane_layer,0] = dep
#         if muti_plane_depth_range[plane_layer,1] == 0 or dep>muti_plane_depth_range[plane_layer,1]:
#             muti_plane_depth_range[plane_layer, 1] = dep


#     image1 = torch.zeros((H,W),device="cuda")
#     image2 = torch.zeros((H,W),device="cuda")
#     x_linspace1 = np.linspace(0+itera_num*int(sample_size/2),H-1,int(H/sample_size)-1-itera_num).astype(int)
#     y_linspace1 = np.linspace(0+itera_num*int(sample_size/2),W-1,int(W/sample_size)-1-itera_num).astype(int)

#     image1[x_linspace1,:] = 1
#     image2[:,y_linspace1] = 1
#     sampler_image = image1+image2

#     x_linspace = torch.linspace(0, W - 1, W).view(1, W).expand(H, W)
#     y_linspace = torch.linspace(0, H - 1, H).view(H, 1).expand(H, W)
#     # xyzs_big = get_pixel_grids(height=H, width=W).permute(1, 0)[:,:2]  # W,H
#     image_axis = torch.stack([x_linspace,y_linspace], dim=2)

#     image_axis = torch.tensor(image_axis,device="cuda",dtype=int)

#     sampler_axis = image_axis[sampler_image==2]
#     sampler_label = plane_mask[sampler_image==2]

#     sampler_depth_t = torch.zeros((sampler_label.shape[0],1),device="cuda")
#     RGB_feature = torch.zeros(sampler_axis.shape[0],3).cuda()

#     K_num = 5

#     for i in range(sampler_axis.shape[0]):
#         layer_label = int(sampler_label[i])

#         RGB_feature[i, :] = image[:, sampler_axis[i, 1], sampler_axis[i, 0]]
#         if muti_mode == "random":
#             sampler_depth_t[i] = muti_plane_depth_range[layer_label,0]+(muti_plane_depth_range[layer_label,1]-muti_plane_depth_range[layer_label,0])*np.random.random()
#         elif muti_mode == "max":
#             sampler_depth_t[i] = muti_plane_depth_range[layer_label,1]
#         elif muti_mode == "neighbor":

#             layer_sample_xy = sampler_xy[sampler_xy_label==layer_label]
#             layer_depth = sampler_depth[sampler_xy_label==layer_label]
#             mh_dis = layer_sample_xy - sampler_axis[i]
#             if mh_dis.shape[0]<=K_num:
#                 mindep = muti_plane_depth_range[layer_label,1]
#             else:
#                 os_dis = torch.sqrt(mh_dis[:, 0] ** 2 + mh_dis[:, 1] ** 2)
#                 indices = torch.tensor(np.argpartition(os_dis.detach().cpu().numpy(), K_num)[:K_num])
#                 sorted_indices = indices[torch.argsort(os_dis[indices])]
#                 mindep = layer_depth[sorted_indices].max()
#             sampler_depth_t[i] = mindep

#     K_invs = torch.inverse(K)
#     R_invs = torch.inverse(R)

#     pts3D = torch.ones((1, 4, sampler_label.shape[0]), device="cuda")


#     pts3D[:, 2:3, :] = sampler_depth_t.unsqueeze(0).permute(0,2,1)
#     pts3D[:, 0:2, :] = (sampler_axis * sampler_depth_t).unsqueeze(0).permute(0,2,1)

#     points = my_view_to_world_coord(pts3D, K_invs, R_invs)
#     SHfeature = RGB2SH(RGB_feature)

#     Init_points = points[0,:3].permute(1,0)


#     return Init_points,SHfeature

def get_depth_mask(render_depth,mono_depth):

    normal_render = render_depth / render_depth.max()
    normal_render = normal_render - normal_render.mean()
    normal_mono = mono_depth / mono_depth.max()
    normal_mono = normal_mono - normal_mono.mean()

    depth_mask = normal_render - normal_mono

    depth_mask[depth_mask < 0] = 0
    depth_mask[depth_mask > 0.15] = 0
    depth_mask[depth_mask != 0] = 1
    # print(depth_mask.shape)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    kernel_erode2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

    mask_erode = cv2.erode(depth_mask, kernel_erode)
    mask_erode1 = cv2.erode(mask_erode, kernel_erode2)

    mask_dilate = cv2.dilate(mask_erode1, kernel_dilate)

    mask_dilate1 = np.array(mask_dilate[:, :, np.newaxis] * 255, dtype=np.uint8)
    contours, _ = cv2.findContours(mask_dilate1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ares = []
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        ares.append(area)


    ares.sort()
    thorld_are = ares[int(len(ares) * 0.4)] + 10

    contour_select = []
    mask_select = np.zeros_like(mask_dilate)
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > thorld_are:
            cv2.drawContours(mask_select, [contour], 0, 1, -1)
            contour_select.append(contour)

    return mask_select

def error_map(render_image,gt_image,static_factor):
    error_map = render_image - gt_image
    error_map = torch.sum(error_map, dim=0)

    top_num = int(error_map.shape[0] * error_map.shape[1] * 0.10)
    error_map1 = error_map.reshape(-1, )

    top_elements = error_map1.topk(top_num)

    element_min = top_elements.values.min()

    error_mask = torch.zeros_like(error_map, dtype=torch.float)
    error_mask[error_map > element_min] = 1

    H, W = error_mask.shape

    # block_mask = torch.zeros_like(error_mask)
    win_size = 16
    if H % win_size != 0:
        H_num = win_size - H % win_size
    else:
        H_num = 0
    if W % win_size != 0:
        W_num = win_size - W % win_size
    else:
        W_num = 0

    padding = (0, W_num, 0, H_num)

    block_mask = torch.nn.functional.pad(error_mask, padding, mode='constant', value=0)
    BH, BW = block_mask.shape
    H_split = int(BH / win_size)
    W_split = int(BW / win_size)
    thrd_num = static_factor * (win_size ** 2)
    for i in range(H_split):
        for j in range(W_split):
            block_sum = torch.sum(
                torch.sum(block_mask[i * win_size:(i + 1) * win_size, j * win_size:(j + 1) * win_size], dim=0), dim=0)
            if block_sum > thrd_num:
                block_mask[i * win_size:(i + 1) * win_size, j * win_size:(j + 1) * win_size] = 1
            else:
                block_mask[i * win_size:(i + 1) * win_size, j * win_size:(j + 1) * win_size] = 0

    return block_mask[:H,:W]



def get_add_point(view_camera, anchor_xyz: torch.Tensor, our_mask: np.ndarray):
    """
    功能：找出 Scaffold-GS 的 3D 锚点中，哪些点投影到了当前视图的 2D 误差区域 (our_mask)。
    
    Args:
        view_camera: 当前视图的相机参数对象。
        anchor_xyz (torch.Tensor): Scaffold-GS 的锚点坐标 (N, 3)，即 gaussians.get_anchor。
        our_mask (np.ndarray): 当前视图的 2D 误差掩码 (H, W)，值 1 表示误差。
        
    Returns:
        np.ndarray: 一个 (N,) 数组，1 表示该锚点被误差区域覆盖，0 否则。
    """
    
    # 1. 准备 世界到相机坐标系 (R) 和 相机内参 (K) 矩阵
    temp_R = copy.deepcopy(view_camera.R)
    temp_T = copy.deepcopy(view_camera.T)

    temp_R = np.transpose(temp_R)
    R = np.eye(4)
    R[:3, :3] = temp_R
    R[:3, 3] = temp_T

    H, W = view_camera.original_image.shape[1], view_camera.original_image.shape[2]
    K = np.eye(4)
    K[0, 2] = view_camera.focal[2] # Cx
    K[1, 2] = view_camera.focal[3] # Cy
    K[0, 0] = view_camera.focal[1] # Fx
    K[1, 1] = view_camera.focal[0] # Fy

    K = torch.FloatTensor(K).unsqueeze(0).cuda()
    R = torch.FloatTensor(R).unsqueeze(0).cuda()

    # 2. 将 3D 锚点坐标转换为齐次坐标 (N, 4) 并投影
    src_xyz_t = anchor_xyz # 使用传入的锚点坐标
    src_xyz_t = src_xyz_t.unsqueeze(0).permute(0, 2, 1) # (1, 3, N)
    tempdata = torch.ones((src_xyz_t.shape[0], 1, src_xyz_t.shape[2])).cuda()
    src_xyz = torch.cat((src_xyz_t, tempdata), dim=1) # (1, 4, N)

    # 假设 world_to_view_screen(src_xyz, R, K) 执行了 K * R * P_world
    xyz_sampler = world_to_view_screen(src_xyz, RT_cam2=R, K=K)

    # 3. 提取 2D 像素坐标
    sampler = xyz_sampler[0, 0:2].transpose(1, 0) # 2D 像素坐标 (u, v), 形状 (N, 2)
    # depth_sampler = xyz_sampler[0, 2:].transpose(1, 0) # 深度 Z，在本函数中未使用

    # 4. 边界检查和钳制
    sampler_t = sampler.detach().cpu().numpy().astype(int)
    
    # ... (原代码中冗余的边界检查/mask部分，因为在下一步的索引钳制中已经处理)
    # 简化：直接对坐标进行钳制
    sampler_t[sampler_t[:, 1] >= H, 1] = H - 1
    sampler_t[sampler_t[:, 0] >= W, 0] = W - 1
    sampler_t[sampler_t < 0] = 0
    # 必须排除投影到边界外的点，以避免索引错误

    # 5. 建立 2D 像素与 3D 锚点索引的映射
    sampler_index = np.arange(sampler_t.shape[0]) # 3D 锚点的原始索引 [0, 1, 2, ..., N-1]

    mask = np.zeros((H,W), dtype=np.int32)
    # 将每个锚点投影到的像素位置的值设置为该锚点的原始索引
    mask[sampler_t[:,1],sampler_t[:,0]] = sampler_index 
    
    # 创建一个 2D 布尔掩码，标记哪些像素有投影点
    mask_temp = np.zeros((H,W))
    mask_temp[mask!=0] = 1

    # 6. 结合外部误差掩码 (our_mask) 进行过滤
    # 只有 '有投影点' (1) 且 '在误差区域' (1) 的像素，相加才等于 2
    mask_temp += our_mask
    mask[mask_temp!=2] = 0 # 仅保留满足条件的像素上的锚点索引

    # 7. 提取最终的锚点索引
    flatten_mask = mask.flatten()
    list_mask = flatten_mask.tolist()
    
    # 排除索引 0 (0 代表没有锚点投影或不在误差区域)
    set_data = set(list_mask)
    select_data = list(set_data)[1:] 

    # 8. 生成输出的 1D 锚点掩码
    Gaussian_mask= np.zeros((anchor_xyz.shape[0]))
    Gaussian_mask[np.array(select_data,dtype=int)] = 1

    return Gaussian_mask

def compute_conv3d(conv3d):
    # print("conv3d",conv3d.shape)
    # complete_conv3d = conv3d[:,:3,:3]
    complete_conv3d = torch.zeros((conv3d.shape[0], 3, 3))
    complete_conv3d[:, 0, 0] = conv3d[:, 0]
    complete_conv3d[:, 1, 0] = conv3d[:, 1]
    complete_conv3d[:, 0, 1] = conv3d[:, 1]
    complete_conv3d[:, 2, 0] = conv3d[:, 2]
    complete_conv3d[:, 0, 2] = conv3d[:, 2]
    complete_conv3d[:, 1, 1] = conv3d[:, 3]
    complete_conv3d[:, 2, 1] = conv3d[:, 4]
    complete_conv3d[:, 1, 2] = conv3d[:, 4]
    complete_conv3d[:, 2, 2] = conv3d[:, 5]

    return complete_conv3d


def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def get_covariance(gaussians,scaling_modifier=1):
    scaling = gaussians.get_scaling
    rotation = gaussians._rotation
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm

def porject_to_2d(viewpoint_camera, points3D):
    full_matrix = viewpoint_camera.full_proj_transform  # w2c @ K
    full_matrix = torch.tensor(full_matrix).cuda()
    # project to image plane
    points3D = F.pad(input=points3D, pad=(0, 1), mode='constant', value=1)
    p_hom = (points3D @ full_matrix).transpose(0, 1)  # N, 4 -> 4, N   -1 ~ 1
    p_w = 1.0 / (p_hom[-1, :] + 0.0000001)
    p_proj = p_hom[:3, :] * p_w

    h = viewpoint_camera.image_height
    w = viewpoint_camera.image_width

    point_image = 0.5 * ((p_proj[:2] + 1) * torch.tensor([w, h]).unsqueeze(-1).to(p_proj.device) - 1) # image plane
    point_image = point_image.detach().clone()
    point_image = torch.round(point_image.transpose(0, 1))

    return point_image

def conv2d_matrix(gaussians, viewpoint_camera, indices_mask, device):
    # 3d convariance matrix
    indices_mask = torch.tensor(indices_mask,dtype=torch.long)
    conv3d = get_covariance(gaussians,scaling_modifier=1)[indices_mask]
    conv3d_matrix = compute_conv3d(conv3d).to(device)

    w2c = viewpoint_camera.world_view_transform
    w2c = torch.tensor(w2c).cuda()
    mask_xyz = gaussians.get_xyz[indices_mask]
    pad_mask_xyz = F.pad(input=mask_xyz, pad=(0, 1), mode='constant', value=1)
    t = pad_mask_xyz @ w2c[:, :3]   # N, 3
    height = viewpoint_camera.image_height
    width = viewpoint_camera.image_width
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_x = width / (2.0 * tanfovx)
    focal_y = height / (2.0 * tanfovy)
    lim_xy = torch.tensor([1.3 * tanfovx, 1.3 * tanfovy]).to(device)
    t[:, :2] = torch.clip(t[:, :2] / t[:, 2, None], -1. * lim_xy, lim_xy) * t[:, 2, None]
    J_matrix = torch.zeros((mask_xyz.shape[0], 3, 3)).to(device)
    J_matrix[:, 0, 0] = focal_x / t[:, 2]
    J_matrix[:, 0, 2] = -1 * (focal_x * t[:, 0]) / (t[:, 2] * t[:, 2])
    J_matrix[:, 1, 1] = focal_y / t[:, 2]
    J_matrix[:, 1, 2] = -1 * (focal_y * t[:, 1]) / (t[:, 2] * t[:, 2])
    W_matrix = w2c[:3, :3]  # 3,3
    T_matrix = (W_matrix @ J_matrix.permute(1, 2, 0)).permute(2, 0, 1) # N,3,3

    conv2d_matrix = torch.bmm(T_matrix.permute(0, 2, 1), torch.bmm(conv3d_matrix, T_matrix))[:, :2, :2]

    return conv2d_matrix

def update(gaussians, view, selected_index, ratios, dir_vector):
    # print("selected_index",selected_index.shape)
    ratios = ratios.unsqueeze(-1).to("cuda")
    selected_xyz = gaussians.get_xyz[selected_index]
    selected_scaling = gaussians.get_scaling[selected_index]
    conv3d = get_covariance(gaussians,scaling_modifier=1)[selected_index]
    conv3d_matrix = compute_conv3d(conv3d).to("cuda")


    eigvals, eigvecs = torch.linalg.eigh(conv3d_matrix)
    max_eigval, max_idx = torch.max(eigvals, dim=1)
    max_eigvec = torch.gather(eigvecs, dim=1,
                              index=max_idx.unsqueeze(1).unsqueeze(2).repeat(1, 1, 3))  # (N, 1, 3)最大特征向量
    long_axis = torch.sqrt(max_eigval) * 3
    max_eigvec = max_eigvec.squeeze(1)
    max_eigvec = max_eigvec / torch.norm(max_eigvec, dim=1).unsqueeze(-1)
    new_scaling = selected_scaling * ratios * 0.8

    max_eigvec_2d = porject_to_2d(view, max_eigvec)
    sign_direction = torch.sum(max_eigvec_2d * dir_vector, dim=1).unsqueeze(-1)
    sign_direction = torch.where(sign_direction > 0, 1, -1)
    new_xyz = selected_xyz + 0.5 * (1 - ratios) * long_axis.unsqueeze(1) * max_eigvec * sign_direction

    selected_index = torch.tensor(selected_index,dtype=torch.long)
    gaussians.reset_xyz(new_xyz,selected_index)
    gaussians.reset_scaling(new_scaling, selected_index)

    # return gaussians

def compute_ratios(conv_2d, points_xy, indices_mask, sam_mask, h, w):
    sam_mask = torch.tensor(sam_mask,dtype=torch.int).cuda()
    indices_mask = torch.tensor(indices_mask,dtype=torch.long)
    means = points_xy[indices_mask]

    eigvals, eigvecs = torch.linalg.eigh(conv_2d)

    max_eigval, max_idx = torch.max(eigvals, dim=1)
    max_eigvec = torch.gather(eigvecs, dim=1,
                        index=max_idx.unsqueeze(1).unsqueeze(2).repeat(1,1,2)) # (N, 1, 2)最大特征向量

    long_axis = torch.sqrt(max_eigval) * 3
    max_eigvec = max_eigvec.squeeze(1)
    max_eigvec = max_eigvec / torch.norm(max_eigvec, dim=1).unsqueeze(-1)
    vertex1 = means + 0.5 * long_axis.unsqueeze(1) * max_eigvec
    vertex2 = means - 0.5 * long_axis.unsqueeze(1) * max_eigvec
    vertex1 = torch.clip(vertex1, torch.tensor([0, 0]).to(points_xy.device), torch.tensor([w-1, h-1]).to(points_xy.device))
    vertex2 = torch.clip(vertex2, torch.tensor([0, 0]).to(points_xy.device), torch.tensor([w-1, h-1]).to(points_xy.device))

    vertex1_xy = torch.round(vertex1).long()
    vertex2_xy = torch.round(vertex2).long()
    vertex1_label = sam_mask[vertex1_xy[:, 1], vertex1_xy[:, 0]]
    vertex2_label = sam_mask[vertex2_xy[:, 1], vertex2_xy[:, 0]]

    index = torch.nonzero(vertex1_label ^ vertex2_label, as_tuple=True)[0]
    special_index = (vertex1_label == 0) & (vertex2_label == 0)
    index = torch.cat((index, special_index), dim=0)

    if index.shape == 0:
        return 0,0,0,0
    elif index.shape[0] != vertex1_xy.shape[0]:
        return 0, 0, 0,0

    selected_vertex1_xy = vertex1_xy[index]
    selected_vertex2_xy = vertex2_xy[index]

    sign_direction = vertex1_label[index] - vertex2_label[index]
    direction_vector = max_eigvec[index] * sign_direction.unsqueeze(-1)

    ratios = []
    update_index = []
    for k in range(len(index)):
        x1, y1 = selected_vertex1_xy[k]
        x2, y2 = selected_vertex2_xy[k]
        # print(k, x1, x2)
        if x1 < x2:
            x_point = torch.arange(x1, x2+1).to(points_xy.device)
            y_point = y1 + (y2- y1) / (x2- x1) * (x_point - x1)
        elif x1 < x2:
            x_point = torch.arange(x2, x1+1).to(points_xy.device)
            y_point = y1 + (y2- y1) / (x2- x1) * (x_point - x1)
        else:
            if y1 < y2:
                y_point = torch.arange(y1, y2+1).to(points_xy.device)
                x_point = torch.ones_like(y_point) * x1
            else:
                y_point = torch.arange(y2, y1+1).to(points_xy.device)
                x_point = torch.ones_like(y_point) * x1

        x_point = torch.tensor(torch.clip(x_point, 0, w-1),dtype=torch.long)
        y_point = torch.tensor(torch.clip(y_point, 0, h-1),dtype=torch.long)


        in_mask = sam_mask[y_point, x_point]

        ratios.append(sum(in_mask) / len(in_mask))

    ratios = torch.tensor(ratios)

    index_in_all = indices_mask[index]

    return index_in_all, ratios, direction_vector,1

def gaussian_decomp(gaussians, viewpoint_camera, input_mask, indices_mask):
    xyz = gaussians.get_xyz
    point_image = porject_to_2d(viewpoint_camera, xyz)

    conv2d = conv2d_matrix(gaussians, viewpoint_camera, indices_mask, device="cuda")
    height = viewpoint_camera.image_height
    width = viewpoint_camera.image_width
    index_in_all, ratios, dir_vector,stage = compute_ratios(conv2d, point_image, indices_mask, input_mask, height, width)

    if stage != 0:
        update(gaussians, viewpoint_camera, index_in_all, ratios, dir_vector)




# 修改
def get_vector(view_cameras, xyz, max_depth, split_num, win_sizes):
    """
    Multi-view anchor relocation for Scaffold-GS.
    Given a list of view_cameras and 3D anchors (N,3), find optimal positions
    by evaluating cross-view SSIM/HSV consistency along depth direction.
    Returns: sampler_xyz [N,3] in world coordinates.
    """

    # --- device & tensor setup ---
    if isinstance(xyz, np.ndarray):
        xyz = torch.from_numpy(xyz).float()
    device = xyz.device if isinstance(xyz, torch.Tensor) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xyz = xyz.to(device=device, dtype=torch.float32)
    assert len(view_cameras) >= 1 and xyz.ndim == 2 and xyz.shape[1] == 3

    # --- reference view setup ---
    view_camera = view_cameras[0]
    temp_R = view_camera.R          # 更正bug.cpu().numpy()
    temp_T = view_camera.T
    R = np.eye(4, dtype=np.float32)
    R[:3, :3] = temp_R.T
    R[:3, 3] = temp_T
    H, W = int(view_camera.original_image.shape[1]), int(view_camera.original_image.shape[2])
    K = np.eye(4, dtype=np.float32)
    K[0, 2], K[1, 2] = W / 2.0, H / 2.0
    K[0, 0], K[1, 1] = float(view_camera.focal[1]), float(view_camera.focal[0])
    K = torch.from_numpy(K).unsqueeze(0).to(device)
    R = torch.from_numpy(R).unsqueeze(0).to(device)

    # --- homogeneous xyz ---
    src_xyz_t = xyz.unsqueeze(0).permute(0, 2, 1)
    ones = torch.ones((1, 1, xyz.shape[0]), device=device)
    src_xyz_h = torch.cat((src_xyz_t, ones), dim=1)

    # --- projection & depth setup ---
    xyz_sampler = world_to_view_screen(src_xyz_h, RT_cam2=R, K=K)
    sampler = xyz_sampler[0, 0:2].transpose(1, 0)
    R_invs = torch.inverse(R)
    K_invs = torch.inverse(K)
    pts3D = torch.ones((1, 4, xyz_sampler.shape[2]), device=device)
    pts3D[:, 2:3, :] = float(max_depth)
    pts3D[:, 0:2, :] = xyz_sampler[:, 0:2, :] * float(max_depth)
    points_world = my_view_to_world_coord(pts3D, K_invs, R_invs)
    depth_vector = (points_world - src_xyz_h) / float(split_num)

    # --- multi-view score accumulation ---
    hsv_weight = torch.tensor([0.2, 0.3, 0.5], device=device)
    scale_weight = torch.tensor([0.4, 0.3, 0.3], device=device)
    all_muti_score = None

    for inx in range(len(view_cameras) - 1):
        ref_cam = view_cameras[inx + 1]
        R_ref = np.eye(4, dtype=np.float32)
        R_ref[:3, :3] = ref_cam.R
        R_ref[:3, 3] = ref_cam.T
        K_ref = np.eye(4, dtype=np.float32)
        K_ref[0, 2], K_ref[1, 2] = W / 2.0, H / 2.0
        K_ref[0, 0], K_ref[1, 1] = float(ref_cam.focal[1]), float(ref_cam.focal[0])
        K_ref = torch.from_numpy(K_ref).unsqueeze(0).to(device)
        R_ref = torch.from_numpy(R_ref).unsqueeze(0).to(device)

        # --- depth sampling ---
        sampler_refs = []
        for i in range(split_num + 1):
            xyz_sample_h = src_xyz_h + float(i) * depth_vector
            xyz_sampler_ref = world_to_view_screen(xyz_sample_h, RT_cam2=R_ref, K=K_ref)
            sampler_ref = xyz_sampler_ref[0, 0:2].transpose(1, 0)
            sampler_refs.append(sampler_ref)

        # --- compute multi-scale scores ---
        muti_score_per_ref = []
        for win_size in win_sizes:
            stable_scale = caculat_muti_scale_SSIM(view_camera, sampler, win_size=win_size)
            scores = []
            for num in range(split_num + 1):
                ref_scale = caculat_muti_scale_SSIM(ref_cam, sampler_refs[num], win_size=win_size)
                if win_size == 1:
                    score = torch.sum(torch.abs(ref_scale - stable_scale) * hsv_weight.unsqueeze(0), dim=1)
                else:
                    score = 1.0 - ssim(ref_scale, stable_scale, window_size=win_size, size_average=False)
                    score = torch.clamp(score, 0.0, 1.0)
                scores.append(score.unsqueeze(1))
            scores = torch.cat(scores, dim=1)
            muti_score_per_ref.append(scores.unsqueeze(2))
        muti_score_per_ref = torch.cat(muti_score_per_ref, dim=2)
        muti_score_per_ref = torch.sum(muti_score_per_ref * scale_weight.view(1, 1, -1), dim=2)

        all_muti_score = muti_score_per_ref if all_muti_score is None else (all_muti_score + muti_score_per_ref)

    # --- final selection ---
    all_muti_score = all_muti_score + 1e-6 * torch.randn_like(all_muti_score)
    _, sample_index = torch.min(all_muti_score, dim=1)
    scale_lins = torch.linspace(0, split_num, split_num + 1, device=device)
    chosen_scales = scale_lins[sample_index].view(1, 1, -1)
    sampler_xyz_h = src_xyz_h + chosen_scales * depth_vector
    sampler_xyz = sampler_xyz_h[0, :3, :].transpose(1, 0).contiguous()

    return sampler_xyz
def caculat_muti_scale_SSIM(view_camera, sampler_t, win_size):
    image = view_camera.original_image
    H, W = image.shape[1], image.shape[2]

    sampler = copy.deepcopy(torch.tensor(sampler_t, dtype=int, device="cuda"))
    sampler[sampler[:, 1] >= H, 1] = H - 1
    sampler[sampler[:, 0] >= W, 0] = W - 1
    sampler[sampler < 0] = 0
    if win_size == 1:
        hsv_image = cv2.cvtColor(image.permute(1, 2, 0).detach().cpu().numpy(), cv2.COLOR_RGB2HSV)
        hsv_image = torch.tensor(hsv_image,device="cuda").permute(2,0,1)
        blocks = torch.zeros([sampler.shape[0], image.shape[0]], device="cuda")
        for i in range(sampler.shape[0]):
            blocks[i, :] = hsv_image[:, sampler[i, 1],sampler[i, 0]]

    else:
        half_win = int(win_size / 2)
        image = torch.nn.functional.pad(image, [half_win, half_win + 1, half_win, half_win + 1])

        blocks = torch.zeros([sampler.shape[0], image.shape[0], win_size, win_size], device="cuda")

        for i in range(sampler.shape[0]):
            blocks[i, :, :, :] = image[:, sampler[i, 1]:sampler[i, 1] + win_size,
                                 sampler[i, 0]:sampler[i, 0] + win_size]

    return blocks
