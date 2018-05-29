from scene_nn import *
import pickle
import pandas as pd
import matplotlib.pyplot as plt

def datasize_vs_loss(overide=False):
    '''

    '''
    if overide:
        tt_acu = pickle.load(open("DATA/datasize_plot_stats.p", "rb"))
        print('finish load datasize_vs_loss data from pickle!')
        return tt_acu
    else:
        # train & test accuracy
        tt_acu = []

        final_data = contextualize(None, True)
        for train_size in reversed(range(10)):
            print('epoch',10-train_size)

            train_size = train_size/10
            print('train_size:',1-train_size)

            train_data, _, test_data = split_data(final_data, train_size=train_size, overide=False)
            tt_data = train(train_data,test_data)
            tt_acu.append(tt_data)
        print('result of cross validation:')
        print(tt_acu)
        pickle.dump(tt_acu, open("DATA/datasize_plot_stats.p", "wb+"))
        print('-success dump datasize_vs_loss data-')
        return tt_acu

def drop_out_vs_loss(overide=False):
    if overide:
        tt_acu = pickle.load(open("DATA/drop_out_plot_stats.p", "rb"))
        print('finish load drop_out_plot_stats data from pickle!')
        return tt_acu
    else:
        # train & test accuracy
        tt_acu = []
        final_data = contextualize(None, True)
        for drop_out in range(1,11):
            print('epoch',drop_out)

            drop_out = drop_out/10
            train_data, _, test_data = split_data(None,0,True)
            tt_data = train(train_data,test_data, drop_out = drop_out)
            tt_acu.append(tt_data)
        print('result of cross validation:')
        print(tt_acu)
        pickle.dump(tt_acu, open("DATA/drop_out_plot_stats.p", "wb+"))
        print('-success dump drop_out_plot_stats data-')
        return tt_acu

def plot(data,xlabel,plot_path):
    '''
    :input data: a list of sublist, one sublist contains data through time of training or cross validation
    '''

    df = pd.DataFrame.from_records(tt_acu, columns=['train','cv'])
    plt.figure();

    df.plot(legend=True)
    plt.xlabel(xlabel)
    plt.ylabel("Loss of the best result")
    plt.savefig(plot_path)

if __name__ == '__main__':
    # plot datasize_vs_loss
    tt_acu = datasize_vs_loss(True)
    plot(tt_acu,"size of dataset","datasize_plot_stats.png")

    # # plot drop_out_vs_loss
    # tt_acu = drop_out_vs_loss()
    # plot(tt_acu,"Regularization term(left after dropout)","drop_out_plot_stats.png")
