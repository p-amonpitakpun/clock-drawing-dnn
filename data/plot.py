import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import sys

import preprocess


def plot_iter(x, y, data=None):
    x_max = np.max(x)
    x_min = np.min(x)
    y_max = np.max(y)
    y_min = np.min(y)

    data_x = x - x_min
    data_y = y_max - y

    tmse = data[person]['TMSE']
    ed = data[person]['ed']
    gender= data[person]['gender']
    side= data[person]['side']
    duration = data[person]['duration']
    diagnosis = data[person]['diagnosis']
    clock_drawing = data[person]['clock_drawing']
    age = data[person]['age']
    # error check
    if data_x.shape != data_y.shape:
        print('both input data must have the same shape!')
        exit(1)

    # set data x, y
    n_frame = data_x.shape[0]

    # prepare a plotting figure
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.set_title('gender:{},side{},age:{},ed:{},TMSE:{}\ndiagnosis:{},clock_drawing:{},duration:{}'.format(gender,side,age,ed,tmse,diagnosis,clock_drawing,duration))
    l  = ax.scatter([],[], c='r', s=1)
    ax.set_ylim(0, y_max - y_min)
    ax.set_xlim(0, x_max - x_min)

    def update(i):

        # update X, Y
        Y = data_y[:i]
        X = data_x[:i]
        l.set_offsets(np.append(X.reshape((-1, 1)), Y.reshape((-1, 1)), axis=1))
        return l, 

    line_ani = animation.FuncAnimation(fig, update, frames=n_frame, interval=10, blit=True, repeat=False)
    plt.show()

    try:
        writer = animation.FFMpegWriter(fps=60, metadata=dict(artist='Me'), bitrate=1800)
        return line_ani, writer
    except:
        return line_ani, None


if __name__ == '__main__':

    argv = sys.argv

    # data = preprocess.load()
    data = preprocess.preprocess(path='data/raw/CD_PD.mat')
    max_person = data.shape[0]

    try:
        person = int(argv[1])
    except:
        person = np.random.randint(0, max_person)

    x = data[person]['x']
    y = data[person]['y']
    t = data[person]['t']
    print('candidate number =', person)

    line_ani, writer = plot_iter(x, y, data)
    try:
        print('saving the video.')
        # line_ani.save('data\\vdo\\clock' + str(person) + '.mp4', writer=writer)
        print('video saved.')
    except Exception as err:
        print('save file unable :', err)
        pass

