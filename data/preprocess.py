from scipy.io import loadmat
import os
import numpy as np

def load(path=os.path.join('data', 'raw', 'CD_PD.mat')):

    return loadmat(path, squeeze_me=True)['dat']

def preprocess(path):

    data = load(path)
    '''
    Refer to the jupyter notebook (Notebook/Data Visualization.ipynb)

    Convert right, left hand to integer -> 0 = left, 1 = right
    '''
    side = data['side']
    side[side == '1-Right'] = 1
    side[side == '0-Left'] = 0

    '''
    Convert male, female to integer -> 0 = male, 1 = female
    '''

    gender = data['gender']
    gender[gender == '1-Female'] = 1
    gender[gender == '0-Male'] = 0

    '''
    Convert education to integer -> (0= no, 1= primary, 2=high school, 3=vocational training, 4=bachelor degree, 5= more than Bachelor)
    '''
    side = data['ed']
    for i in range(len(side)):
        side[i] = int(side[i][0])
    
    '''
    Replace age = 0 with mean age.
    '''
    age = data['age']
    age[age == 0] = np.mean(age[age != 0])

    '''
    Replace TMSE = 0 with the specified rules:
    <4 ปี อาจใช้เกณฑ์ 19 คะแนน
    5 - 8 ปี อาจใช้เกณฑ์ 23 คะแนน
    9 - 12 ปีอาจใช้เกณฑ์ 27 คะแนน
    ระดับอุดมศึกษา อาจใช้เกณฑ์ 29 คะแนน
    '''
    diagnosis = data['diagnosis']
    for i in range(len(diagnosis)):
        diagnosis[i] = 1-diagnosis[i]
    tmse = data['TMSE']
    for i in range(len(tmse)):
        if tmse[i] == 0:
            if data['age'][i] < 4:
                tmse[i] = 19
            elif data['age'][i] <= 8:
                tmse[i] = 23
            elif data['age'][i] <= 12:
                tmse[i] = 27
            else:
                tmse[i] = 29
    return data

def split_by_patient(data, ratio=0.8, randomize=True):

    '''
    This function will split data by ratio in range [0, 1] by choosing randomly, and then return 2 patient index numpy arrays.
    ratio : a real number ranging between 0 and 1, to specify the ratio of the partition of train, test
    randomize : if True, then splits the data randomly else, split by index
    '''

    bound = int(ratio*len(data))
    if randomize:
        idx_train = np.random.choice(len(data), bound , replace=False)
        idx_test = np.array([e for e in np.arange(len(data)) if e not in idx_train])
    else:
        arr = np.arange(len(data))
        idx_train = arr[:bound]
        idx_test = arr[bound:]
#     print(idx_train, idx_test)

    return data[idx_train], data[idx_test]

def get_age_gender_data(data, mode='diagnosis'):
    #get shuffled data
    data_train, data_test = split_by_patient(data)
    
    #label PD
    y_test = data_test[mode].astype(np.float32)
    y_train = data_train[mode].astype(np.float32)
    
    age_train = data_train['age'].astype(np.float32)
    age_train = age_train.reshape(1, *age_train.shape).T
    gender_train = data_train['gender'].astype(np.float32)
    gender_train = gender_train.reshape(1, *gender_train.shape).T

    age_test = data_test['age'].astype(np.float32)
    age_test = age_test.reshape(1, *age_test.shape).T
    gender_test = data_test['gender'].astype(np.float32)
    gender_test = gender_test.reshape(1, *gender_test.shape).T
    
    #x
    x_train = np.concatenate((age_train,gender_train),axis=1)
    x_test = np.concatenate((age_test, gender_test), axis=1)
    
    return x_test, x_train, y_test, y_train


def get_all_non_temporal_data(data, mode='diagnosis'):
    #get shuffled data
    data_train, data_test = split_by_patient(data)
    
    #label PD
    y_test = data_test[mode].astype(np.float32)
    y_train = data_train[mode].astype(np.float32)
    
    features = ['age', 'gender', 'ed', 'side', 'TMSE']
    features_train, features_test = [], []
    for f in features:
        feature_train = data_train[f].astype(np.float32)
        feature_train = feature_train.reshape(1, *data_train.shape).T
        feature_test = data_test[f].astype(np.float32)
        feature_test = feature_test.reshape(1, *data_test.shape).T
        
        features_train.append(feature_train)
        features_test.append(feature_test)
    #x
    x_train = np.concatenate(features_train,axis=1)
    x_test = np.concatenate(features_test , axis=1)
    
    return x_test, x_train, y_test, y_train


def get_image(X, Y, shape=(100, 100)):
    
    try:
        h, w = shape
    except:
        raise Exception('error : get_image() -> an argument shape must be (h, w).')
    
    x_min, x_max = min(X), max(X)
    y_min, y_max = min(Y), max(Y)
    delta_x = x_max - x_min
    delta_y = y_max - y_min
    ratio_xy = delta_x / delta_y
    ratio_wh = w / h
    
    scale = 1
    dx, dy = 0, 0
    
    if ratio_xy > ratio_wh:
        s = (w - 1) / delta_x
        dy = int((h / 2) - (delta_y * s / 2))
    else:
        s = (h - 1) / delta_y
        dx = int((w /2 ) - (delta_x * s / 2))
    
#     print('1', type(scale), type(dx), type(dy))
    img = np.zeros((h, w))
#     print(type(img), img.shape)
    for x, y in zip(X, Y):
        x0 = x - x_min
        y0 = y - y_min
        x_int = int(x0 * s + dx)
        y_int = int(y0 * s + dy)
#         print('0', type(x_int), type(y_int))
        try:
            img[y_int][x_int] = 1
        except IndexError:
            index_str = f'\n\tshape {shape} index {(y_int, x_int)}'
            scale_str = f' \n\twith scale {s} delta {(dx, dy)}'
            raise IndexError(f'error: get_image -> index out of bound' + index_str + scale_str)

#     vertical flip to visualize better
#     img = img[::-1,:]
    return img

# test get_image
def test_get_image():
    dat = preprocess('data\\raw\\CD_PD.mat')
    X = dat[0]['x']
    Y = dat[0]['y']
    img = get_image(X, Y, 200, 200)
    import cv2
    cv2.imshow('', img)
    cv2.waitKey()
        

# test get_image
def test_get_image():
    dat = preprocess('data\\raw\\CD_PD.mat')
    X = dat[0]['x']
    Y = dat[0]['y']
    img = get_image(X, Y, 200, 200)
    import cv2
    cv2.imshow('', img)
    cv2.waitKey()
        

if __name__ == '__main__':

    data = load()
    print(data.shape)

    # test_get_image()
    
