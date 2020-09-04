import os

import matplotlib.pyplot as plt
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE
BUFFER_SIZE = 60000
BATCH_SIZE = 32

IMG_HEIGHT = 96
IMG_WIDTH = 96


def load_and_preprocess_image(path):
    """
    加载指定路径的图像
    :param path:
    :return:
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_WIDTH, IMG_HEIGHT])
    image = (image - 127.5) / 127.5
    return image


class AnimeAvatarDatasets(object):

    def __init__(self, datasets_path, batch_size=BATCH_SIZE):
        self.batch_size = batch_size

        print("GalaProfile 数据集位置：" + os.path.realpath(datasets_path))  # 输出数据集所在目录
        print("--")

        # 统计训练图像数量
        # self.train_data_total = len(os.listdir(os.path.realpath(datasets_path)))
        # print('数据集中图像数量：', self.train_data_total)

        # 为文件夹中所有 *.jpg 文件建立索引
        all_images_tensor = tf.data.Dataset.list_files(datasets_path + '/*.jpg')

        # 加载图像并预处理构成 train_data
        self.train_data = all_images_tensor.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
        self.train_data = self.train_data.shuffle(BUFFER_SIZE).batch(batch_size=self.batch_size)


if __name__ == '__main__':
    dataset = AnimeAvatarDatasets('../二次元妹子头像数据集/datasets')

    plt.figure(figsize=(8, 8))
    for n, image in enumerate(dataset.train_data.take(4)):
        plt.subplot(2, 2, n + 1)
        plt.imshow(image)
        plt.axis('off')
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
    plt.show()
