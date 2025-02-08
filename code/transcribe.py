import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
import os
from os import path
import glob


class Transcribe():
    
    def __init__(self,
                 voxel_size,
                 image_size,
                 dataname,
                 start,
                 stop,
                 ran,
                 rotated_pcd_input_path,
                 predict_input_path,
                 vector_input_path,
                 output_path
                 ):
        self.voxel_size = voxel_size
        self.voxel_radius = voxel_size*math.sqrt(3)/2
        self.image_size = image_size
        self.dataname = dataname
        self.start = start
        self.stop = stop
        self.ran = ran
        self.rotated_pcd_input_path = rotated_pcd_input_path
        self.predict_input_path = predict_input_path
        self.vector_input_path = vector_input_path
        self.output_path = output_path


    def load(self, index, i, vector_file, start=0):
        image = np.array(Image.open(f'{self.predict_input_path}/{self.dataname}_{index}/{self.dataname}_{index}_{start+i:05}.png'))[:,:,0:3]
        rotated_pcd = np.load(f'{self.rotated_pcd_input_path}/{self.dataname}_{index}/{self.dataname}_{index}_{start+i:05}.npy')[:, 0:3]
        vector = vector_file[i, :]
        return image, rotated_pcd, vector


    def transcript(self, image, pcd):
        xy = np.array(pcd[:,0:2])
        scaled_xy = (xy + self.voxel_radius)*self.image_size/(2*self.voxel_radius) #scale from root(3)^2 to 224^2
        r_xy = np.floor(scaled_xy).astype(int) #translate coordinates to array indices
        transcribed_color = [image[r_xy[i,1], r_xy[i,0], :] for i in range(pcd.shape[0])] #transcribe color to every point
        transcribed_pcd = np.concatenate([pcd, transcribed_color], 1)
        return transcribed_pcd


    def rotate(self, transcribed_pcd, vector):
        normal_vector = vector[0:3]
        voxel_center = vector[3:6]
        z_average = vector[6]
        transcribed_pcd[:,2] = transcribed_pcd[:,2] + z_average #add the average to z coordinates
        
        #first rotation of normal vector
        theta1 = math.atan2(normal_vector[2], normal_vector[0])
        phi1 = math.pi/2 - theta1
        
        rotate_matrix1 = [[math.cos(phi1), -math.sin(phi1)], [math.sin(phi1), math.cos(phi1)]]
        normal_matrix1 = [normal_vector[0], normal_vector[2]]
        [normal_vector[0], normal_vector[2]] = np.dot(rotate_matrix1, normal_matrix1)

        #second rotation of normal vector
        theta2 = math.atan2(normal_vector[2], normal_vector[1])
        phi2 = math.pi/2 - theta2
        
        rotate_matrix2 = [[math.cos(phi2), -math.sin(phi2)], [math.sin(phi2), math.cos(phi2)]]
        normal_matrix2 = [normal_vector[1], normal_vector[2]]
        [normal_vector[1], normal_vector[2]] = np.dot(rotate_matrix2, normal_matrix2)

        #inversion of rotation matrix
        inv_rotate_matrix2 = np.linalg.inv(rotate_matrix2)
        inv_rotate_matrix1 = np.linalg.inv(rotate_matrix1)
        
        #rotate pointcloud
        for i in range(transcribed_pcd.shape[0]):
            pcd_matrix2 = [transcribed_pcd[i,1], transcribed_pcd[i,2]]
            [transcribed_pcd[i,1], transcribed_pcd[i,2]] = np.dot(inv_rotate_matrix2, pcd_matrix2)
            pcd_matrix1 = [transcribed_pcd[i,0], transcribed_pcd[i,2]]
            [transcribed_pcd[i,0], transcribed_pcd[i,2]] = np.dot(inv_rotate_matrix1, pcd_matrix1)

        #parallel traslation
        transcribed_pcd[:,0:3] = transcribed_pcd[:,0:3] + voxel_center

        return transcribed_pcd


    def conduct(self):
        for index in range(int(self.start), int(self.stop+self.ran), int(self.ran)):
            index = float(index)
            pcd = np.empty([0, 6])
            vector_file = np.load(f'{self.vector_input_path}{self.dataname}_{index}/{self.dataname}_{index}.npy')
            files = len(glob.glob(f'{self.predict_input_path}/{self.dataname}_{index}/*.png'))
            for i in range(files):
                image, rotated_pcd, vector= self.load(index, i, vector_file)
                transcribed_pcd = self.transcript(image, rotated_pcd)
                transcribed_pcd = self.rotate(transcribed_pcd, vector)
                pcd = np.concatenate([pcd, transcribed_pcd])
            os.makedirs(f'{self.output_path}', exist_ok=True)
            np.save(f'{self.output_path}{self.dataname}_{index}', pcd)


def main():
    ts = Transcribe(voxel_size=1.0, image_size=224, dataname="Esashito-PointCloud", start=83158.0, stop=83158.0, ran=2.0,
                    rotated_pcd_input_path = "rotated_pcd/",
                    predict_input_path = "predicted_image/",
                    vector_input_path = "vector/",
                    output_path = "transcribed_pcd/"
                    )
    ts.conduct()

    
if __name__ == '__main__':
    main()