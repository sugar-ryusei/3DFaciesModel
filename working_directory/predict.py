import numpy as np
import cv2
import os
import glob
import tensorflow as tf


class Predict(object):

    def __init__(self,
                image_size,
                dataname,
                start,
                stop,
                ran,
                batch_size,
                model_name,
                palette,
                input_path,
                output_path,
                model_direc
                ):
        self.image_size = image_size
        self.dataname = dataname
        self.start = start
        self.stop = stop
        self.ran = ran
        self.batch_size = batch_size
        self.model_name = model_name
        self.palette = palette
        self.input_path = input_path
        self.output_path = output_path
        self.model_direc = model_direc

    def load_dataset(self, direc):
        data_paths = glob.glob(f'{self.input_path}{direc}/*.npy')
        self.data_paths = [os.path.splitext(os.path.basename(path))[0] for path in data_paths]
        data = [np.load(data_file).astype(np.float32) for data_file in data_paths]#
        data = self.normalize_z(data)
        data = [self.normalize_color(i) for i in data]
        ds = tf.data.Dataset.from_tensor_slices(data)
        ds = ds.map(
            self.load_data,
            num_parallel_calls=tf.data.AUTOTUNE
            )
        ds = ds.batch(self.batch_size)
        return ds
    
    def load_data(self, data):
        return data

    def normalize_z(self, data):
        ran_max = max([np.max(i[:,:,0]) for i in data])
        ran_min = min([np.min(i[:,:,0]) for i in data])
        ran = ran_max - ran_min
        mid_ran = (ran_max + ran_min)/2
        data = [self.normalize_temp(i, ran, mid_ran) for i in data]
        return data

    def normalize_temp(self, data, ran, mid_ran):
        data[:,:,0] = (data[:,:,0]/ran*2) - mid_ran/ran*2
        return data

    def normalize_color(self, image):
        image[:,:,-3:] = (image[:,:,-3:]/127.5) - 1
        return image

    def predict(self, data):
        model = tf.keras.models.load_model(f'{self.model_direc}{self.model_name}.h5')
        pred = model.predict(data)
        return pred

    def color_conversion(self, c, i):
        color_value = self.palette[c][i]        
        return color_value
    
    def generate_image(self, array):
        array = np.array(array).reshape(50176,)
        r_value = [self.color_conversion(c, 0) for c in array]
        g_value = [self.color_conversion(c, 1) for c in array]
        b_value = [self.color_conversion(c, 2) for c in array]

        array = np.stack([b_value, g_value, r_value], 1)
        image = array.reshape(224,224,3)
        return image

    def translate(self, pred):
        predicted_class = pred.argmax(axis=3).tolist()
        images = [self.generate_image(array) for array in predicted_class]
        return images

    def save(self, images, direc):
        os.makedirs(f'{self.output_path}{direc}', exist_ok=True)
        for i in range(len(images)):
            tmp_image_path = f'{self.output_path}{direc}/{self.data_paths[i]}.png'
            image = np.array(images[i])
            cv2.imwrite(tmp_image_path, image)

    def conduct(self):
        for index in range(int(self.start), int(self.stop+self.ran), int(self.ran)):
            index = float(index)
            direc = f'{self.dataname}_{index}'
            data = self.load_dataset(direc)
            pred = self.predict(data)
            images = self.translate(pred)
            self.save(images, direc)


def main():
    pd = Predict(image_size=224,
                 dataname="Esashito-PointCloud",
                 start=83158.0,
                 stop=83158.0,
                 ran=2.0,
                 batch_size=8,
                 model_name='resunet',
                 palette = np.array([[255,108,0],
                                   [255,228,0],
                                   [0,255,0],
                                   [97,57,14],
                                   [0,24,255],
                                   [255,255,255]]),
                 input_path = "array/",
                 output_path = "predicted_image/",
                 model_direc = "training_result/")
    pd.conduct()

if __name__ == '__main__':
    main()