import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import math
import cv2
from scipy.interpolate import Rbf
import os
from os import path
import gc


class DatasetGeneration():
    
    def __init__(self,
                 voxel_size,
                 radius_for_normal,
                 normal_count,
                 image_size,
                 bit_depth,
                 z_threshold,
                 dataname,
                 start,
                 stop,
                 ran,
                 input_path,
                 array_output_path,
                 image_output_path,
                 vector_output_path,
                 rotated_pcd_output_path
                 ):
        self.voxel_size = voxel_size
        self.voxel_radius = voxel_size*math.sqrt(3)/2
        self.radius_for_normal = radius_for_normal
        self.normal_count = normal_count
        self.image_size = image_size
        self.bit_depth = bit_depth
        self.z_threshold = z_threshold
        self.input_path = input_path
        self.dataname = dataname
        self.start = start
        self.stop = stop
        self.ran = ran
        self.array_output_path = array_output_path
        self.image_output_path = image_output_path
        self.vector_output_path = vector_output_path
        self.rotated_pcd_output_path = rotated_pcd_output_path


    def load(self, index):
        loadpcd = np.load(f'{self.input_path}{self.dataname}_{index}.npy')
        points = loadpcd[:,0:3]
        colors = loadpcd[:,3:6]
        print("loaded pcd with a shape of ", loadpcd.shape)
        return points, colors


    def make_pcd(self, points, colors):
        o3dpcd = o3d.geometry.PointCloud()
        o3dpcd.points = o3d.utility.Vector3dVector(points)
        o3dpcd.colors = o3d.utility.Vector3dVector(colors/(2**self.bit_depth-1))
        pcd_tree = o3d.geometry.KDTreeFlann(o3dpcd)

        #estimation of normal vector of points
        o3dpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.radius_for_normal, max_nn=self.normal_count))

        #downsample point cloud
        o3ddownpcd = o3dpcd.voxel_down_sample(voxel_size=self.voxel_size)

        #convert to ndarray
        downpcd = np.asarray(o3ddownpcd.points)
        normal = np.asarray(o3ddownpcd.normals)
        return downpcd, normal, o3dpcd, pcd_tree


    def extract(self, o3dpcd, downpcd, normal, voxel_number, pcd_tree):
        #coordinates of voxel center
        voxel_center = downpcd[voxel_number,:]

        #find neighboring points with distance less than the radius
        [k, idx, _] = pcd_tree.search_radius_vector_3d(voxel_center, self.voxel_radius)

        #extract neighboring points and colors
        extracted_pcd = np.asarray(o3dpcd.points)[idx, :]
        extracted_color = np.asarray(o3dpcd.colors)[idx, :]
        normal_vector = normal[voxel_number]
        
        #integrate duplicate points into one point
        all = np.concatenate([extracted_pcd, extracted_color], 1)
        all = all.tolist()
        all = sorted(all)
        all = np.array(all)
        k=0
        for i in range(all.shape[0]):
            i = i+k
            try:
                if all[i,0]==all[i+1,0] and all[i,1]==all[i+1,1] and all[i,2]==all[i+1,2]:
                    all[i,3] = (all[i,3]+all[i+1,3])/2
                    all[i,4] = (all[i,4]+all[i+1,4])/2
                    all[i,5] = (all[i,5]+all[i+1,5])/2
                    all = np.delete(all, i+1, 0)
                    k = k-1
            except:
                break

        extracted_pcd = all[:,0:3]
        extracted_color = all[:,3:6]        
        return extracted_pcd, normal_vector, voxel_center, extracted_color


    #pcd and plane rotated to be parallel to xy plane
    def rotate(self, extracted_pcd, normal_vector, voxel_center):
        original_normal_vector = np.array([normal_vector[0], normal_vector[1], normal_vector[2]])
        original_pcd = extracted_pcd - voxel_center
        
        #first rotation-----------------------------------------------------------------

        #set the angles from normal vector
        theta1 = math.atan2(normal_vector[2], normal_vector[0])
        phi1 = math.pi/2 - theta1
        #first rotation of normal vector
        rotate_matrix1 = [[math.cos(phi1), -math.sin(phi1)], [math.sin(phi1), math.cos(phi1)]]
        normal_matrix1 = [normal_vector[0], normal_vector[2]]
        [normal_vector[0], normal_vector[2]] = np.dot(rotate_matrix1, normal_matrix1)
  
        #second rotation----------------------------------------------------------------

        #set the angles from normal vector
        theta2 = math.atan2(normal_vector[2], normal_vector[1])
        phi2 = math.pi/2 - theta2

        #second rotation of normal vector
        rotate_matrix2 = [[math.cos(phi2), -math.sin(phi2)], [math.sin(phi2), math.cos(phi2)]]
        normal_matrix2 = [normal_vector[1], normal_vector[2]]
        [normal_vector[1], normal_vector[2]] = np.dot(rotate_matrix2, normal_matrix2)

        for i in range(extracted_pcd.shape[0]):
            pcd_matrix1 = [original_pcd[i,0], original_pcd[i,2]]
            [original_pcd[i,0], original_pcd[i,2]] = np.dot(rotate_matrix1, pcd_matrix1)
            pcd_matrix2 = [original_pcd[i,1], original_pcd[i,2]]
            [original_pcd[i,1], original_pcd[i,2]] = np.dot(rotate_matrix2, pcd_matrix2)

        #normalize by average of z coordinates
        z_average = np.average(original_pcd[:,2])
        original_pcd[:,2] = original_pcd[:,2] - z_average

        rotated_pcd = original_pcd

        vector = [original_normal_vector[0], original_normal_vector[1], original_normal_vector[2],
                  voxel_center[0], voxel_center[1], voxel_center[2],
                  z_average]

        return rotated_pcd, vector
        
    
    #interpolate z coordinate and colors
    def interpolate(self, rotated_pcd, extracted_color):
        x = rotated_pcd[:,0]
        y = rotated_pcd[:,1]
        z = rotated_pcd[:,2]

        r = extracted_color[:,0]
        g = extracted_color[:,1]
        b = extracted_color[:,2]
        
        xm = np.linspace(-self.voxel_radius, self.voxel_radius, self.image_size)
        ym = np.linspace(-self.voxel_radius, self.voxel_radius, self.image_size)
        xx, yy = np.meshgrid(xm, ym)

        rbf_z = Rbf(x, y, z, function='gaussian')
        rbf_z = rbf_z(xx, yy)
        rbf_r = Rbf(x, y, r, function='gaussian')
        rbf_r = rbf_r(xx, yy)
        rbf_g = Rbf(x, y, g, function='gaussian')
        rbf_g = rbf_g(xx, yy)
        rbf_b = Rbf(x, y, b, function='gaussian')
        rbf_b = rbf_b(xx, yy)

        array = np.stack([rbf_z, rbf_b, rbf_g, rbf_r], axis=2)
        array = self.normalize(array)
        image = array[:,:,1:4]

        return array, image
    
    
    #normalize
    def normalize(self, array):
        z = array[:,:,0]
        colors = array[:,:,1:4]

        #normalize z
        z = np.where(z<-self.z_threshold, -self.z_threshold, z)
        z = np.where(z>self.z_threshold, self.z_threshold, z)
        z = z.reshape([self.image_size, self.image_size, 1])

        #normalize colors
        colors = np.where(colors<0, 0, colors)
        colors = np.where(colors>1, 1, colors)
        colors = colors*255

        array = np.concatenate([z, colors], 2)

        return array
    
    
    #save_images as .png files
    def save_image(self, image, index, i, start=0):
        os.makedirs(f'{self.image_output_path}{self.dataname}_{index}', exist_ok=True)
        tmp_image_path = f'{self.image_output_path}{self.dataname}_{index}/{self.dataname}_{index}_{start+i:05}.png'
        cv2.imwrite(tmp_image_path, image)

    #save z coordintes and colors as .npy files (training data)
    def save_array(self, array, index, i, start=0):
        os.makedirs(f'{self.array_output_path}{self.dataname}_{index}', exist_ok=True)
        tmp_array_path = f'{self.array_output_path}{self.dataname}_{index}/{self.dataname}_{index}_{start+i:05}'
        np.save(tmp_array_path, array)

    #save original normal vectors for every images
    def save_vector(self, vectors, index):
        os.makedirs(f'{self.vector_output_path}{self.dataname}_{index}', exist_ok=True)
        tmp_vector_path = f'{self.vector_output_path}{self.dataname}_{index}/{self.dataname}_{index}'
        np.save(tmp_vector_path, vectors)

    #save rotated point cloud coordinates
    def save_rotated_pcd(self, rotated_pcd, extracted_color, index, i, start=0):
        rotated_pcd = np.concatenate([rotated_pcd, extracted_color], 1)
        os.makedirs(f'{self.rotated_pcd_output_path}{self.dataname}_{index}', exist_ok=True)
        tmp_rotated_pcd_path = f'{self.rotated_pcd_output_path}{self.dataname}_{index}/{self.dataname}_{index}_{start+i:05}'
        np.save(tmp_rotated_pcd_path, rotated_pcd)
        
    
    def conduct(self):
        for index in range(int(self.start), int(self.stop+self.ran), int(self.ran)):
            index = float(index)
            points, colors = self.load(index)
            downpcd, normal, o3dpcd, pcd_tree = self.make_pcd(points, colors)
            vectors = np.empty([0,7])

            print(f"making images of {index}th point cloud")
            for i in range(downpcd.shape[0]):
                extracted_pcd, normal_vector, voxel_center, extracted_color = self.extract(o3dpcd, downpcd, normal, i, pcd_tree)
                rotated_pcd, vector = self.rotate(extracted_pcd, normal_vector, voxel_center)
                self.save_rotated_pcd(rotated_pcd, extracted_color, index, i)
                vectors = np.vstack([vectors, vector])
                
                array, image = self.interpolate(rotated_pcd, extracted_color)
                self.save_array(array, index, i)
                self.save_image(image, index, i)

            self.save_vector(vectors, index)

            del points, colors
            gc.collect()


def main():
    dg = DatasetGeneration(voxel_size=1.0,
                           radius_for_normal=0.5,
                           normal_count=50000,
                           image_size=224,
                           bit_depth=16,
                           z_threshold = 0.2,
                           dataname="Esashito-PointCloud",
                           start=83158.0,
                           stop=83158.0,
                           ran=2.0,
                           input_path = "original_data/",
                           array_output_path = "array/",
                           image_output_path = "image/",
                           vector_output_path = "vector/",
                           rotated_pcd_output_path = "rotated_pcd/"
                           )
    dg.conduct()
    
    
if __name__ == '__main__':
    main()