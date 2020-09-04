import os

import matplotlib.pyplot as plt
import tensorflow as tf

NOISE_DIM = 100  # 噪声向量维度

num_examples_to_generate = 16

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)  # 计算交叉熵的辅助函数


def generator():
    """
    生成器
    生成器使用 tf.keras.layers.Conv2DTranspose 转置卷积层来从随机噪声中产生图片
    输入的随机噪声为 (100维)

    tf.keras.layers.Dense(
        units: 神经元数目（输出空间的维度）,
        activation: 指定激活函数,
        use_bias: 是否使用偏置矢量
    )

    转置卷积：越卷积图像越大
    tf.keras.layers.Conv2DTranspose(
        filters: 卷积核数目（输出空间的维数）,
        kernel_size: 卷积核 size,
        strides: 卷积的步长,
        padding: valid / same,
        use_bias: 是否使用偏置矢量
    )

    每次转置卷积输出数据的 shape: (a, b, c)
    卷积核数目决定最终输出的 c 的值
    卷积核大小与步长决定 a、b 的值
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(3 * 3 * 1024, use_bias=False, input_shape=(NOISE_DIM,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((3, 3, 1024)))
    assert model.output_shape == (None, 3, 3, 1024)

    # 第一次反卷积
    model.add(tf.keras.layers.Conv2DTranspose(512, (4, 4), strides=(1, 1), padding='valid', use_bias=False))
    assert model.output_shape == (None, 6, 6, 512)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    # 第二次反卷积
    model.add(tf.keras.layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 12, 12, 256)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    # 第三次反卷积
    model.add(tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 24, 24, 128)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    # 第四次反卷积
    model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 48, 48, 64)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 96, 96, 3)
    return model


def discriminator():
    """
    判别器

    一个基于 CNN 的图像分类器
    为真实图片输出正值，为伪造图片输出负值
    """
    dropout = 0.4
    model = tf.keras.Sequential([
        # 第一层卷积，卷积核 64 个，大小 (4, 4)
        tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding="valid", input_shape=(96, 96, 3)),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        # tf.keras.layers.Dropout(dropout),

        # 第二层卷积，卷积核 128 个，大小 (4, 4)
        tf.keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding="same"),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dropout(dropout),

        # 第三层卷积，卷积核 256 个，大小 (4, 4)
        tf.keras.layers.Conv2D(256, (4, 4), strides=(2, 2), padding="same"),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dropout(dropout),

        # 第四层卷积，卷积核 512 个，大小 (4, 4)
        tf.keras.layers.Conv2D(512, (4, 4), strides=(2, 2), padding="same"),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dropout(dropout),

        # 第五层卷积，卷积核 512 个，大小 (5, 5)
        tf.keras.layers.Conv2D(512, (4, 4), strides=(2, 2), padding="same"),
        tf.keras.layers.LeakyReLU(alpha=0.2),
        tf.keras.layers.Dropout(dropout),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1)
    ])
    return model


def generator_loss_fun(fake_output):
    """
    生成器损失函数

    生成器的目标是生成的图像经判别器鉴定得到的结果为 1
    越接近 1 说明生成的图像效果越好
    因此，其损失为判别器鉴定结果与全 1 tensor 的 交叉熵
    """
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def discriminator_loss_fun(real_output, fake_output):
    """
    判别器损失函数

    判别器的目标是对真实图像的鉴定结果为 1，对伪造图像的鉴定结果为 0
    因此，分别计算对真实图像的鉴定结果与全 1 tensor 的交叉熵，对伪造图像的鉴定结果与全 0 tensor 的交叉熵
    两者的和为判别器的损失
    """
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


class DCGAN(object):
    """
    DCGAN

    深度卷积生成对抗网络
    """
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)  # 生成器的优化器
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)  # 判别器的优化器

    def __init__(self, checkpoint_dir='./training_checkpoints'):
        self.discriminator = discriminator()
        self.generator = generator()

        # 设置检查点
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator
        )
        # 载入检查点
        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    @tf.function
    def train_step(self, real_images, batch_size):
        """
        自定义训练循环
        """
        noise = tf.random.normal([batch_size, NOISE_DIM])  # 生成随机噪声

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)  # 生成器利用随机噪声生成伪造图像
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            # 计算损失
            generator_loss = generator_loss_fun(fake_output)
            discriminator_loss = discriminator_loss_fun(real_output, fake_output)

            # 根据损失计算梯度
            gradients_of_generator = gen_tape.gradient(generator_loss, self.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(discriminator_loss, self.discriminator.trainable_variables)

        # 应用梯度下降优化网络权重
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def train(self, datasets, epochs):
        """
        训练函数
        """
        seed = tf.random.normal([num_examples_to_generate, NOISE_DIM])  # 生成随机噪声

        for epoch in range(epochs):
            print("Epoch:", epoch + 1)
            for input_images in datasets.train_data:
                self.train_step(input_images, batch_size=datasets.batch_size)

            self.generate_and_save_images(epoch=epoch + 1, test_input=seed)  # 生成并保存图像
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)  # 保存检查点

    def generate_and_save_images(self, epoch, test_input):
        """
        生成及保存图像
        """
        prediction = self.generator(test_input, training=False)

        plt.figure(figsize=(8, 8))
        for i in range(prediction.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(prediction[i])
            plt.axis('off')
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
        plt.savefig('./generated/image_at_epoch_{:04d}.png'.format(epoch))
        plt.close()

    def generate_one(self):
        seed = tf.random.normal([num_examples_to_generate, NOISE_DIM])  # 生成随机噪声
        prediction = self.generator(seed, training=False)

        plt.figure(figsize=(8, 8))
        for i in range(prediction.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(prediction[i])
            plt.axis('off')
        plt.show()
        plt.close()


if __name__ == '__main__':
    seed = tf.random.normal([num_examples_to_generate, NOISE_DIM])

    model = DCGAN()
    prediction = model.generator(seed, training=False)
    fake_output = model.discriminator(prediction, training=False)

    print("判别器:", fake_output)

    plt.figure(figsize=(8, 8))
    for i in range(prediction.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(prediction[i])
        plt.axis('off')
    plt.show()
