from scipy.io import loadmat


def load(path="data\\raw\\CD_PD.mat"):

    return loadmat(path, squeeze_me=True)['dat']


if __name__ == '__main__':

    data = load()
    print(data.shape)
