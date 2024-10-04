import torch
import numpy as np
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    NormWeightedCompositor,
)
from typing import Tuple, Optional
from pytorch3d import transforms as torch3d_tf
from scipy.spatial.transform import Rotation
import utils.math3d as math3d


def denorm_rgb(x):
    v = (x + 1) / 2 * 255.0
    return v.to(torch.uint8)


def flatten_img_pc_to_points(obs, pcd):
    bs = obs[0][0].shape[0]
    # concatenating the points from all the cameras
    # (bs, num_points, 3)
    pc = torch.cat([p.permute(0, 2, 3, 1).reshape(bs, -1, 3) for p in pcd], 1)
    _img_feat = [o[0] for o in obs]
    img_dim = _img_feat[0].shape[1]
    # (bs, num_points, 3)
    img_feat = torch.cat([p.permute(0, 2, 3, 1).reshape(bs, -1, img_dim) for p in _img_feat], 1)
    img_feat = (img_feat + 1) / 2
    return pc, img_feat


def clamp_pc_in_bound(pc, img_feat, bounds, skip=False):
    if skip: return pc, img_feat
    x_min, y_min, z_min, x_max, y_max, z_max = bounds
    inv_pnt = ( # invalid points
        (pc[:, :, 0] < x_min)
        | (pc[:, :, 0] > x_max)
        | (pc[:, :, 1] < y_min)
        | (pc[:, :, 1] > y_max)
        | (pc[:, :, 2] < z_min)
        | (pc[:, :, 2] > z_max)
        | torch.isnan(pc[:, :, 0])
        | torch.isnan(pc[:, :, 1])
        | torch.isnan(pc[:, :, 2])
    )
    # TODO: move from a list to a better batched version
    pc = [pc[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)]
    if img_feat is not None:
        img_feat = [img_feat[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)]
    return pc, img_feat



def place_pc_in_cube(
    pc, app_pc=None, with_mean_or_bounds=True, scene_bounds=None
):
    """
    calculate the transformation that would place the point cloud (pc) inside a
        cube of size (2, 2, 2). The pc is centered at mean if with_mean_or_bounds
        is True. If with_mean_or_bounds is False, pc is centered around the mid
        point of the bounds. The transformation is applied to point cloud app_pc if
        it is not None. If app_pc is None, the transformation is applied on pc.
    :param pc: pc of shape (num_points_1, 3)
    :param with_mean_or_bounds:
        Either:
            True: pc is centered around its mean
            False: pc is centered around the center of the scene bounds
    :param scene_bounds: [x_min, y_min, z_min, x_max, y_max, z_max]
    """
    if with_mean_or_bounds:
        assert scene_bounds is None
    else:
        assert scene_bounds is not None
    if with_mean_or_bounds:
        pc_mid = (torch.max(pc, 0)[0] + torch.min(pc, 0)[0]) / 2
        x_len, y_len, z_len = torch.max(pc, 0)[0] - torch.min(pc, 0)[0]
    else:
        x_min, y_min, z_min, x_max, y_max, z_max = scene_bounds
        pc_mid = torch.tensor(
            [
                (x_min + x_max) / 2,
                (y_min + y_max) / 2,
                (z_min + z_max) / 2,
            ]
        ).to(pc.device)
        x_len, y_len, z_len = x_max - x_min, y_max - y_min, z_max - z_min
    scale = 2 / max(x_len, y_len, z_len)
    if app_pc is None:
        app_pc = torch.clone(pc)
    app_pc = (app_pc - pc_mid) * scale
    # reverse transformation to obtain app_pc in original frame
    def rev_trans(x):
        return (x / scale) + pc_mid
    return app_pc, rev_trans


def transform_pc(pc, loc, sca):
    """
    change location of the center of the pc and scale it
    :param pc:
        either:
        - tensor of shape(b, num_points, 3)
        - tensor of shape(b, 3)
        - list of pc each with size (num_points, 3)
    :param loc: (b, 3 )
    :param sca: 1 or (3)
    """
    assert len(loc.shape) == 2
    assert loc.shape[-1] == 3
    if isinstance(pc, list):
        assert all([(len(x.shape) == 2) and (x.shape[1] == 3) for x in pc])
        pc = [sca * (x - y) for x, y in zip(pc, loc)]
    elif isinstance(pc, torch.Tensor):
        assert len(pc.shape) in [2, 3]
        assert pc.shape[-1] == 3
        if len(pc.shape) == 2:
            pc = sca * (pc - loc)
        else:
            pc = sca * (pc - loc.unsqueeze(1))
    else:
        assert False

    # reverse transformation to obtain app_pc in original frame
    def rev_trans(x):
        assert isinstance(x, torch.Tensor)
        return (x / sca) + loc

    return pc, rev_trans



def generate_heatmap_from_screen_pts(pt, res, sigma, thres_sigma_times=3):
    """
    Pytorch code to generate heatmaps from point. Points with values less than
    thres are made 0
    :type pt: torch.FloatTensor of size (num_pt, 2)
    :type res: int or (int, int)
    :param sigma: the std of the gaussian distribition. if it is -1, we
        generate a hm with one hot vector
    :type sigma: float
    :type thres: float
    """
    num_pt, x = pt.shape
    assert x == 2
    assert sigma > 0

    if isinstance(res, int):
        resx = resy = res
    else:
        resx, resy = res

    _hmx = torch.arange(0, resy).to(pt.device)
    _hmx = _hmx.view([1, resy]).repeat(resx, 1).view([resx, resy, 1])
    _hmy = torch.arange(0, resx).to(pt.device)
    _hmy = _hmy.view([resx, 1]).repeat(1, resy).view([resx, resy, 1])
    hm = torch.cat([_hmx, _hmy], dim=-1)
    hm = hm.view([1, resx, resy, 2]).repeat(num_pt, 1, 1, 1) # one HxW heatmap for each point?

    pt = pt.view([num_pt, 1, 1, 2])
    hm = torch.exp(-1 * torch.sum((hm - pt) ** 2, -1) / (2 * (sigma**2))) # RBF Kernel
    thres = np.exp(-1 * (thres_sigma_times**2) / 2) # truncated
    hm[hm < thres] = 0.0

    hm /= torch.sum(hm, (1, 2), keepdim=True) + 1e-6 # normalization
    return hm # (n_pt, h, w)



def preprocess_images_in_batch(sample_dict, camera_keys):
    def _norm_rgb(x):
        return (x.float() / 255.0) * 2.0 - 1.0

    obs, pcds = [], []
    for n in camera_keys:
        rgb = sample_dict["%s_rgb" % n]
        pcd = sample_dict["%s_point_cloud" % n].float()

        rgb = _norm_rgb(rgb).float()

        obs.append(
            [rgb, pcd]
        )  # obs contains both rgb and pointcloud (used in ARM for other baselines)
        pcds.append(pcd)  # only pointcloud
    return obs, pcds


def get_cube_R_T(
    with_scale=False, cameras=['top', 'front', 'back', 'left', 'right']
):
    """
    Returns camera rotations and translations (in batched) to render point cloud around a cube
    """
    elev_azim = {
        "top": (0, 0),
        "front": (90, 0),
        "back": (270, 0),
        "left": (0, 90),
        "right": (0, 270),
    }

    elev = torch.tensor([elev_azim[c][0] for c in cameras])
    azim = torch.tensor([elev_azim[c][1] for c in cameras])

    up = []
    dist = []
    scale = []
    for view in cameras:
        if view in ["left", "right"]:
            up.append((0, 0, 1))
        else:
            up.append((0, 1, 0))

        dist.append(1)
        scale.append((1, 1, 1))

    # https://matplotlib.org/stable/api/toolkits/mplot3d/view_angles.html
    # about look_at transformation

    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, up=up)
    out = [R, T]
    if with_scale:
        out.append(scale)
    return out


# source: https://discuss.pytorch.org/t/batched-index-select/9115/6
def batched_index_select(inp, dim, index):
    """
    input: B x * x ... x *
    dim: 0 < scalar
    index: B x M
    """
    views = [inp.shape[0]] + [1 if i != dim else -1 for i in range(1, len(inp.shape))]
    expanse = list(inp.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(inp, dim, index)


def grid_sample_from_heatmap(
    screen_points: torch.Tensor, heatmap: torch.Tensor, points_weight: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor]:
    """
    :param screen_points: 
        continuous location of point coordinates from where value needs to be
        selected. it is of size [nc, npt, 2], locations in pytorch3d screen
        notations
    :param heatmap: size [nc, 1, h, w]
    :param points_weight:
        some predifined weight of size [nc, npt], it is used along with the
        distance weights
    :return:
        - the weighted average for each point according to the hm values. the size is [nc, npt, 1]. 
        - the second and third elements are intermediate values to be used while caching
    """
    nc, nw, h, w = heatmap.shape
    if points_weight is None:
        npt = screen_points.shape[1]
        points_weight = torch.ones([nc, npt]).to(heatmap.device)

        # giving points outside the image zero weight
        points_weight[screen_points[:, :, 0] < 0] = 0
        points_weight[screen_points[:, :, 1] < 0] = 0
        points_weight[screen_points[:, :, 0] > (w - 1)] = 0
        points_weight[screen_points[:, :, 1] > (h - 1)] = 0

        screen_points = screen_points.unsqueeze(2).repeat([1, 1, 4, 1])
        # later used for calculating weight
        screen_points_copy = screen_points.detach().clone()

        # getting discrete grid location of pts in the camera image space
        screen_points[:, :, 0, 0] = torch.floor(screen_points[:, :, 0, 0])
        screen_points[:, :, 0, 1] = torch.floor(screen_points[:, :, 0, 1])
        screen_points[:, :, 1, 0] = torch.floor(screen_points[:, :, 1, 0])
        screen_points[:, :, 1, 1] = torch.ceil(screen_points[:, :, 1, 1])
        screen_points[:, :, 2, 0] = torch.ceil(screen_points[:, :, 2, 0])
        screen_points[:, :, 2, 1] = torch.floor(screen_points[:, :, 2, 1])
        screen_points[:, :, 3, 0] = torch.ceil(screen_points[:, :, 3, 0])
        screen_points[:, :, 3, 1] = torch.ceil(screen_points[:, :, 3, 1])
        grid_screen_points = screen_points.long()  # [nc, npt, 4, 2] (grid)

        # o─────────────o
        # │             │
        # │   x         │
        # │             │
        # │             │
        # │             │
        # │             │
        # │             │
        # o─────────────o
        # since we are taking modulo, points at the edge, i,e at h or w will be
        # mapped to 0. this will make their distance from the continous location
        # large and hence they won't matter. therefore we don't need an explicit
        # step to remove such points
        grid_screen_points[:, :, :, 0] = torch.fmod(grid_screen_points[:, :, :, 0], int(w))
        grid_screen_points[:, :, :, 1] = torch.fmod(grid_screen_points[:, :, :, 1], int(h))
        grid_screen_points[grid_screen_points < 0] = 0

        # getting normalized weight for each discrete location for pt
        # weight based on distance of point from the discrete location
        # [nc, npt, 4]
        screen_points_dist = 1 / (torch.sqrt(torch.sum((screen_points_copy - grid_screen_points) ** 2, dim=-1)) + 1e-10)
        points_weight = points_weight.unsqueeze(-1) * screen_points_dist
        _points_weight = torch.sum(points_weight, dim=-1, keepdim=True)
        _points_weight[_points_weight == 0.0] = 1
        # cached screen_points_wei in select_feat_from_hm_cache
        points_weight = points_weight / _points_weight  # [nc, npt, 4]

        grid_screen_points = grid_screen_points.view(nc, 4 * npt, 2)  # [nc, 4 * npt, 2]
        # cached screen_points in select_feat_from_hm_cache
        grid_screen_points = (grid_screen_points[:, :, 1] * w) + grid_screen_points[:, :, 0]  # [nc, 4 * npt]
    else:
        grid_screen_points = screen_points

    # transforming indices from 2D to 1D to use pytorch gather
    heatmap = heatmap.permute(0, 2, 3, 1).view(nc, h * w, nw)  # [nc, h * w, nw]
    # [nc, 4 * npt, nw]
    screen_points_val = batched_index_select(heatmap, dim=1, index=grid_screen_points)
    # tranforming back each discrete location of point
    screen_points_val = screen_points_val.view(nc, -1, 4, nw)
    # summing weighted contribution of each discrete location of a point
    screen_points_val = torch.sum(screen_points_val * points_weight.unsqueeze(-1), dim=2) # [nc, npt, nw]
    return screen_points_val, grid_screen_points, points_weight




class CubePointCloudRenderer:
    """
    Can be used to render point clouds with fixed cameras and dynamic cameras.
    Code flow: Given the arguments we create a fixed set of cameras around the
    object. We can then render camera images (__call__()), project 3d
    points on the camera images (get_pt_loc_on_img()), project heatmap
    featues onto 3d points (get_feat_frm_hm_cube()) and get 3d point with the
    max heatmap value (get_max_3d_frm_hm_cube()).

    For each of the functions, we can either choose to use the fixed cameras
    which were defined by the argument or create dynamic cameras. The dynamic
    cameras are created by passing the dyn_cam_info argument (explained in
    _get_dyn_cam()).

    For the fixed camera, we optimize the code for projection of heatmap and
    max 3d calculation by caching the values in self._pts, self._fix_pts_cam
    and self._fix_pts_cam_wei.
    """

    def __init__(
        self,
        device,
        img_size, # (h, w)
        radius=0.012, 
        points_per_pixel=5,
        with_depth=False,
        R_T_scale=None,
        cameras=['top', 'front', 'back', 'left', 'right']
    ):
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        h, w = img_size
        assert h == w
        if isinstance(device, (int, str)):
            device = torch.device(device)
        self.device = device
        self.img_size = img_size
        self.radius = radius
        self.points_per_pixel = points_per_pixel
        self.with_depth = with_depth

        # render settings
        self.raster_settings = PointsRasterizationSettings(
            image_size=self.img_size,
            radius=self.radius,
            points_per_pixel=self.points_per_pixel,
            bin_size=0,
        )
        self.points_compositor = NormWeightedCompositor()
        if R_T_scale is not None:
            R, T, scale = R_T_scale
        else:
            R, T, scale = get_cube_R_T(
                with_scale=True, cameras=cameras
            )
        assert len(R.shape) == len(T.shape) + 1 == 3
        self.cameras = FoVOrthographicCameras(
            device=self.device,
            R=R,
            T=T,
            znear=0.01,
            scale_xyz=scale,
        )
        self.num_imgs = self.num_cameras = len(R)

        self.rasterizer = PointsRasterizer(
                cameras=self.cameras, raster_settings=self.raster_settings
        )

        self.anchor_points3d = None

        self.cached_grid_screen_points = None
        self.cached_screen_points_weight = None

    def post_process_normalized_image(self, imgs):
        """
        some post processing of the images
        """
        if self.with_depth:
            if imgs[..., :-1].max() > 1:
                assert imgs[..., :-1].max() < 1.001, imgs[..., :-1].max()
                imgs[..., :-1] /= imgs[..., :-1].max()
        else:
            if imgs.max() > 1:
                assert imgs.max() < 1.001
                imgs /= imgs.max()
        return imgs

    @torch.no_grad()
    def __call__(self, pc, feat):
        """
        :param pc: torch.Tensor  (num_point, 3)
        :param feat: torch.Tensor (num_point, num_feat)
        :return: (num_img,  h, w, num_feat)
        """
        assert pc.shape[-1] == 3
        assert len(pc.shape) == 2
        assert len(feat.shape) == 2
        assert isinstance(pc, torch.Tensor)

        pc = [pc]
        feat = [feat]
        img = []

        p3d_pc = Pointclouds(points=pc, features=feat)
        p3d_pc = p3d_pc.extend(self.num_cameras)
        fragments = self.rasterizer(p3d_pc)
        if self.with_depth:
            # meaning of zbuf: https://github.com/facebookresearch/pytorch3d/issues/1147
            depth = fragments.zbuf[..., 0]
            _, h, w = depth.shape # (num_img, h, w)
            depth_0 = depth == -1
            depth_sum = torch.sum(depth, (1, 2)) + torch.sum(depth_0, (1, 2))
            depth_mean = depth_sum / ((h * w) - torch.sum(depth_0, (1, 2)))
            depth -= depth_mean.unsqueeze(-1).unsqueeze(-1)
            depth[depth_0] = -1
        # Construct weights based on the distance of a point to the true point.
        # However, this could be done differently: e.g. predicted as opposed
        # to a function of the weights.
        r = self.rasterizer.raster_settings.radius
        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.points_compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            p3d_pc.features_packed().permute(1, 0),
        )
        # permute so image channels at the end
        images = images.permute(0, 2, 3, 1)
        if self.with_depth:
            images = torch.cat((images, depth.unsqueeze(-1)), dim=-1)
        images = self.post_process_normalized_image(images)
        return images

    @torch.no_grad()
    def points3d_to_screen2d(self, pt):
        """
        returns the location of a point on the image of the cameras
        :param pt: torch.Tensor of shape (bs, np, 3)
        :returns: the location of the pt on the image. this is different from the
            camera screen coordinate system in pytorch3d. the difference is that
            pytorch3d camera screen projects the point to [0, 0] to [H, W]; while the
            index on the img is from [0, 0] to [H-1, W-1]. We verified that
            the to transform from pytorch3d camera screen point to img we have to
            subtract (1/H, 1/W) from the pytorch3d camera screen coordinate.
        :return type: torch.Tensor of shape (bs, np, num_cameras, 2)
        """
        assert len(pt.shape) == 3
        assert pt.shape[-1] == 3
        bs, np = pt.shape[0:2]
        # (num_cameras, bs * np, 2)
        pt_scr = self.cameras.transform_points_screen(
            pt.view(-1, 3), image_size=self.img_size
        )[..., 0:2]
        pt_scr = torch.transpose(pt_scr, 0, 1)
        # transform from camera screen to image index
        h, w = self.img_size
        # (bs * np, num_cameras, 2)
        pt_scr = pt_scr - torch.tensor((1 / w, 1 / h)).to(pt_scr.device)
        pt_scr = pt_scr.view(bs, np, self.num_cameras, 2)
        return pt_scr

    @torch.no_grad()
    def get_anchor_points3d_heatmap(self, heatmap):
        """
        :param hm: torch.Tensor of (1, num_cameras, h, w)
        :return: tupe of ((num_cameras, h^3, 1), (h^3, 3))
        """
        x, nc, h, w = heatmap.shape
        assert x == 1
        assert nc == self.num_cameras
        assert self.img_size == (h, w)

        if self.anchor_points3d is None:
            pts = torch.linspace(-1 + (1 / h), 1 - (1 / h), h).to(heatmap.device)
            self.anchor_points3d = torch.cartesian_prod(pts, pts, pts)

        if self.cached_screen_points_weight is None:
            # (nc, np, 2)
            self.cached_grid_screen_points = self.points3d_to_screen2d(self.anchor_points3d.unsqueeze(0)).squeeze(0).transpose(0, 1)
        
        # (nc, np, bs)
        anchor_points_heatmap, self.cached_grid_screen_points, self.cached_screen_points_weight = grid_sample_from_heatmap(
            self.cached_grid_screen_points, heatmap.transpose(0, 1)[0 : self.num_cameras], self.cached_screen_points_weight
        )
        return anchor_points_heatmap, self.anchor_points3d

    @torch.no_grad()
    def get_most_likely_point_3d(self, hm):
        """
        given set of heat maps, return the 3d location of the point with the
            largest score, assumes the points are in a cube [-1, 1]. This function
            should be used  along with the render. For standalone version look for
            the other function with same name in the file.
        :param hm: (1, nc, h, w)
        :return: (1, 3)
        """
        x, nc, h, w = hm.shape
        assert x == 1
        assert nc == self.num_cameras
        assert self.img_size == (h, w)

        pts_hm, anchor_pts3d = self.get_anchor_points3d_heatmap(hm)
        # (bs, np, nc)
        pts_hm = pts_hm.permute(2, 1, 0)
        # (bs, np)
        pts_hm = torch.mean(pts_hm, -1)
        # (bs)
        ind_max_pts = torch.argmax(pts_hm, -1)
        return anchor_pts3d[ind_max_pts]
    
    
    def reset(self):
        # del self.anchor_points3d
        self.cached_grid_screen_points = None
        self.cached_screen_points_weight = None
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()


def rand_dist(size, min=-1.0, max=1.0):
    return (max - min) * torch.rand(size) + min


def perturb_se3(pcd, trans_shift_4x4, rot_shift_4x4, action_gripper_4x4, bounds):
    """Perturb point clouds with given transformation.
    :param pcd:
        Either:
        - list of point clouds [[bs, 3, H, W], ...] for N cameras
        - point cloud [bs, 3, H, W]
        - point cloud [bs, 3, num_point]
        - point cloud [bs, num_point, 3]
    :param trans_shift_4x4: translation matrix [bs, 4, 4]
    :param rot_shift_4x4: rotation matrix [bs, 4, 4]
    :param action_gripper_4x4: original keyframe action gripper pose [bs, 4, 4]
    :param bounds: metric scene bounds [bs, 6]
    :return: peturbed point clouds in the same format as input
    """
    # batch bounds if necessary

    # for easier compatibility
    single_pc = False
    if not isinstance(pcd, list):
        single_pc = True
        pcd = [pcd]

    bs = pcd[0].shape[0]
    if bounds.shape[0] != bs:
        bounds = bounds.repeat(bs, 1)

    perturbed_pcd = []
    for p in pcd:
        p_shape = p.shape
        permute_p = False
        if len(p.shape) == 3:
            if p_shape[-1] == 3:
                num_points = p_shape[-2]
                p = p.permute(0, 2, 1)
                permute_p = True
            elif p_shape[-2] == 3:
                num_points = p_shape[-1]
            else:
                assert False, p_shape

        elif len(p.shape) == 4:
            assert p_shape[-1] != 3, p_shape[-1]
            assert p_shape[-2] != 3, p_shape[-2]
            num_points = p_shape[-1] * p_shape[-2]

        else:
            assert False, len(p.shape)

        action_trans_3x1 = (
            action_gripper_4x4[:, 0:3, 3].unsqueeze(-1).repeat(1, 1, num_points)
        )
        trans_shift_3x1 = (
            trans_shift_4x4[:, 0:3, 3].unsqueeze(-1).repeat(1, 1, num_points)
        )

        # flatten point cloud
        p_flat = p.reshape(bs, 3, -1)
        p_flat_4x1_action_origin = torch.ones(bs, 4, p_flat.shape[-1]).to(p_flat.device)

        # ! shift points to have action_gripper pose as the origin
        # ! because the augmentation is action-centric
        p_flat_4x1_action_origin[:, :3, :] = p_flat - action_trans_3x1

        # ! apply rotation
        perturbed_p_flat_4x1_action_origin = torch.bmm(
            p_flat_4x1_action_origin.transpose(2, 1), rot_shift_4x4
        ).transpose(2, 1)

        # apply bounded translations
        bounds_x_min, bounds_x_max = bounds[:, 0].min(), bounds[:, 3].max()
        bounds_y_min, bounds_y_max = bounds[:, 1].min(), bounds[:, 4].max()
        bounds_z_min, bounds_z_max = bounds[:, 2].min(), bounds[:, 5].max()

        action_then_trans_3x1 = action_trans_3x1 + trans_shift_3x1
        action_then_trans_3x1_x = torch.clamp(
            action_then_trans_3x1[:, 0], min=bounds_x_min, max=bounds_x_max
        )
        action_then_trans_3x1_y = torch.clamp(
            action_then_trans_3x1[:, 1], min=bounds_y_min, max=bounds_y_max
        )
        action_then_trans_3x1_z = torch.clamp(
            action_then_trans_3x1[:, 2], min=bounds_z_min, max=bounds_z_max
        )
        action_then_trans_3x1 = torch.stack(
            [action_then_trans_3x1_x, action_then_trans_3x1_y, action_then_trans_3x1_z],
            dim=1,
        )
        # ! shift back the origin + perturbation
        perturbed_p_flat_3x1 = perturbed_p_flat_4x1_action_origin[:, :3, :] + action_then_trans_3x1
        if permute_p:
            perturbed_p_flat_3x1 = torch.permute(perturbed_p_flat_3x1, (0, 2, 1))
        perturbed_p = perturbed_p_flat_3x1.reshape(p_shape)
        perturbed_pcd.append(perturbed_p)
    if single_pc:
        perturbed_pcd = perturbed_pcd[0]
    return perturbed_pcd


def apply_se3_augmentation(
    pcd,
    center,
    bounds,
    trans_aug_range,
    rot_aug_range,
):
    """Apply SE3 augmentation to a point clouds and actions.
    :param pcd: [bs, num_points, 3]
    :param action_gripper_pose: 6-DoF pose of keyframe action [bs, 7]
    :param bounds: metric scene bounds
        Either:
        - [bs, 6]
        - [6]
    :param trans_aug_range: range of translation augmentation
        [x_range, y_range, z_range]; this is expressed as the percentage of the scene bound
    :param rot_aug_range: range of rotation augmentation [x_range, y_range, z_range]
    :return: perturbed action_gripper_pose,  pcd
    """

    # batch size
    bs = pcd.shape[0]
    device = pcd.device

    if len(bounds.shape) == 1:
        bounds = bounds.unsqueeze(0).repeat(bs, 1).to(device)
    if len(trans_aug_range.shape) == 1:
        trans_aug_range = trans_aug_range.unsqueeze(0).repeat(bs, 1).to(device)
    if len(rot_aug_range.shape) == 1:
        rot_aug_range = rot_aug_range.unsqueeze(0).repeat(bs, 1)

    # identity matrix
    identity_4x4 = torch.eye(4).unsqueeze(0).repeat(bs, 1, 1).to(device=device)

    # 4x4 matrix of keyframe action gripper pose
    action_gripper_trans = center

    # applying gimble fix to calculate a new action_gripper_rot
    action_gripper_4x4 = identity_4x4.detach().clone()
    action_gripper_4x4[:, 0:3, 3] = action_gripper_trans


    # sample translation perturbation with specified range
    # augmentation range is a percentage of the scene bound
    trans_range = (bounds[:, 3:] - bounds[:, :3]) * trans_aug_range.to(device=device)
    # rand_dist samples value from -1 to 1
    trans_shift = trans_range * rand_dist((bs, 3)).to(device=device)

    # apply bounded translations
    bounds_x_min, bounds_x_max = bounds[:, 0], bounds[:, 3]
    bounds_y_min, bounds_y_max = bounds[:, 1], bounds[:, 4]
    bounds_z_min, bounds_z_max = bounds[:, 2], bounds[:, 5]

    # ! ensure action will not be saturated
    trans_shift[:, 0] = torch.clamp(
        trans_shift[:, 0],
        min=bounds_x_min - action_gripper_trans[:, 0],
        max=bounds_x_max - action_gripper_trans[:, 0],
    )
    trans_shift[:, 1] = torch.clamp(
        trans_shift[:, 1],
        min=bounds_y_min - action_gripper_trans[:, 1],
        max=bounds_y_max - action_gripper_trans[:, 1],
    )
    trans_shift[:, 2] = torch.clamp(
        trans_shift[:, 2],
        min=bounds_z_min - action_gripper_trans[:, 2],
        max=bounds_z_max - action_gripper_trans[:, 2],
    )

    trans_shift_4x4 = identity_4x4.detach().clone()
    trans_shift_4x4[:, 0:3, 3] = trans_shift

    roll = np.deg2rad(rot_aug_range[:, 0:1].cpu() * rand_dist((bs, 1)))
    pitch = np.deg2rad(rot_aug_range[:, 1:2].cpu() * rand_dist((bs, 1)))
    yaw = np.deg2rad(rot_aug_range[:, 2:3].cpu() * rand_dist((bs, 1)))
    rot_shift_3x3 = torch3d_tf.euler_angles_to_matrix(
        torch.cat((roll, pitch, yaw), dim=1), "XYZ"
    ).to(device=device)
    rot_shift_4x4 = identity_4x4.detach().clone()
    rot_shift_4x4[:, :3, :3] = rot_shift_3x3

    # ! perturbed actions
    perturbed_action_gripper_4x4 = identity_4x4.detach().clone()
    perturbed_action_gripper_4x4[:, 0:3, 3] = action_gripper_4x4[:, 0:3, 3]
    perturbed_action_gripper_4x4[:, :3, :3] = torch.bmm(
        rot_shift_4x4.transpose(1, 2)[:, :3, :3], action_gripper_4x4[:, :3, :3]
    )
    perturbed_action_gripper_4x4[:, 0:3, 3] += trans_shift

    # convert transformation matrix to translation + quaternion
    perturbed_action_trans = perturbed_action_gripper_4x4[:, 0:3, 3].cpu().numpy()
    # apply perturbation to pointclouds
    # takes care for not moving the point out of the image
    pcd = perturb_se3(pcd, trans_shift_4x4, rot_shift_4x4, action_gripper_4x4, bounds)
    return pcd, perturbed_action_trans



def add_uniform_noise(x, u):
    """
    adds uniform noise to a tensor x. output is tensor where each element is
    in [x-u, x+u]
    :param x: tensor
    :param u: float
    """
    assert isinstance(u, float)
    # move noise in -1 to 1
    noise = (2 * torch.rand(*x.shape, device=x.device)) - 1
    x = x + (u * noise)
    return x
