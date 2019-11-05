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



if __name__ == '__main__':

    data = load()
    print(data.shape)
