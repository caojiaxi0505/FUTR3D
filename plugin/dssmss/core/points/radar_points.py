from mmdet3d.core.points.base_points import BasePoints
import torch


class RadarPoints(BasePoints):
    def __init__(self, tensor, points_dim=3, attribute_dims=None):
        super(RadarPoints, self).__init__(
            tensor, points_dim=points_dim, attribute_dims=attribute_dims
        )
        self.rotation_axis = 2

    def flip(self, bev_direction="horizontal"):
        """Flip the boxes in BEV along given BEV direction."""
        if bev_direction == "horizontal":
            self.tensor[:, 1] = -self.tensor[:, 1]
            self.tensor[:, 4] = -self.tensor[:, 4]
        elif bev_direction == "vertical":
            self.tensor[:, 0] = -self.tensor[:, 0]
            self.tensor[:, 3] = -self.tensor[:, 3]

    def scale(self, scale_factor):
        self.tensor[:, :3] *= scale_factor
        self.tensor[:, 3:5] *= scale_factor

    def rotate(self, rotation, axis=None):
        if not isinstance(rotation, torch.Tensor):
            rotation = self.tensor.new_tensor(rotation)
        assert (
            rotation.shape == torch.Size([3, 3]) or rotation.numel() == 1
        ), f"invalid rotation shape {rotation.shape}"
        if axis is None:
            axis = self.rotation_axis
        if rotation.numel() == 1:
            rot_sin = torch.sin(rotation)
            rot_cos = torch.cos(rotation)
            if axis == 1:
                rot_mat_T = rotation.new_tensor(
                    [[rot_cos, 0, -rot_sin], [0, 1, 0], [rot_sin, 0, rot_cos]]
                )
            elif axis == 2 or axis == -1:
                rot_mat_T = rotation.new_tensor(
                    [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]]
                )
            elif axis == 0:
                rot_mat_T = rotation.new_tensor(
                    [[0, rot_cos, -rot_sin], [0, rot_sin, rot_cos], [1, 0, 0]]
                )
            else:
                raise ValueError("axis should in range")
            rot_mat_T = rot_mat_T.T
        elif rotation.numel() == 9:
            rot_mat_T = rotation
        else:
            raise NotImplementedError
        self.tensor[:, :3] = self.tensor[:, :3] @ rot_mat_T
        self.tensor[:, 3:5] = self.tensor[:, 3:5] @ rot_mat_T[:2, :2]
        return rot_mat_T

    def in_range_bev(self, point_range):
        in_range_flags = (
            (self.tensor[:, 0] > point_range[0])
            & (self.tensor[:, 1] > point_range[1])
            & (self.tensor[:, 0] < point_range[2])
            & (self.tensor[:, 1] < point_range[3])
        )
        return in_range_flags

    def convert_to(self, dst, rt_mat=None):
        from mmdet3d.core.bbox import Coord3DMode

        return Coord3DMode.convert_point(
            point=self, src=Coord3DMode.LIDAR, dst=dst, rt_mat=rt_mat
        )
