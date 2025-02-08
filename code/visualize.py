import numpy as np
import os
import open3d as o3d


class Visualize(object):

    def __init__(self,
                 bit_depth,
                 down_sample_rate,
                 x_adjustment,
                 y_adjustment,
                 dataname,
                 index,
                 input_path
                ):
        
        self.bit_depth = bit_depth
        self.down_sample_rate = down_sample_rate
        self.x_adjustment = x_adjustment
        self.y_adjustment = y_adjustment
        self.input_path = (f"{input_path}{dataname}_{index}.npy")

    def load(self):
        pcd = np.load(self.input_path)
        points = pcd[:, 0:3]
        colors = pcd[:, 3:6]

        #adjustment of x and y coordinates
        print("points are distributed around", points[0:5,:])
        points[:,0] = points[:,0] + self.x_adjustment
        points[:,1] = points[:,1] + self.y_adjustment
        return points, colors


    def visualize(self, points, colors):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors/(2**self.bit_depth-1))
        pcd = pcd.uniform_down_sample(every_k_points=self.down_sample_rate)
        o3d.visualization.draw_geometries([pcd])


    def color_check(self, colors):
        im = np.dot(colors[:,0], 1000000) + np.dot(colors[:,1], 1000) + colors[:,2]
        im_u, con = np.unique(im, return_counts=True)
        num_classes = len(im_u)
        print(im_u)
        print("number of colors:", num_classes)


    def conduct(self):
        points, colors = self.load()
        self.color_check(colors)
        self.visualize(points, colors)


def main():
    vs = Visualize(bit_depth=8,
                   down_sample_rate=1,
                   x_adjustment=-8.3369e+04,
                   y_adjustment=9.1965e+04,
                   dataname="Esashito-PointCloud",
                   index=83158.0,
                   input_path="transcribed_pcd/")
    vs.conduct()

if __name__ == '__main__':
    main()