import sys

from dcgan import DCGAN
from anime_avatar_datasets import AnimeAvatarDatasets


def train():
    datasets = AnimeAvatarDatasets('..\\二次元妹子头像数据集\\datasets', batch_size=32)
    model = DCGAN()
    model.train(datasets, epochs=64)


def generate_one():
    model = DCGAN()
    model.generate_one()


if __name__ == '__main__':
    if sys.argv[1] == 't':
        train()
    elif sys.argv[1] == 'g':
        generate_one()
    else:
        print("Error")
