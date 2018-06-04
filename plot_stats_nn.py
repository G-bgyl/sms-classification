from scene_nn import *
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
#  plot loss over different size of dataset
# ------------------------------
def datasize_vs_loss(overide=False):
    '''

    '''
    unit = 20
    if overide:
        tt_acu = pickle.load(open("DATA/datasize_loss_stats_clustered.p", "rb"))
        print('finish load datasize_vs_loss data from pickle!')
        return tt_acu
    else:
        # train & test accuracy
        tt_acu = []

        # final_data = contextualize(None, True)

        print('Begin training for datasize_vs_loss!')
        for train_size in reversed(range(unit)):
            print('epoch', unit -train_size)
            train_size = train_size / unit
            print('train_size:',1-train_size)
            # train_size in split data means the part that need to throw away
            train_data, cv_data, _ = split_data(final_data, train_size=train_size, overide=False)
            train_loss_, cv_loss_,train_f1_score,cv_f1_score = train(train_data,cv_data)
            tt_acu.append([1-train_size, train_loss_, cv_loss_,train_f1_score,cv_f1_score])

        print('result of cross validation:')
        print(tt_acu)
        pickle.dump(tt_acu, open("DATA/datasize_loss_stats_clustered.p", "wb+"))
        print('-success dump datasize_vs_loss data-')
        return tt_acu
# ------------------------------
# plot loss of different dropout
# ------------------------------
def drop_out_vs_loss(overide=False):
    '''
    : input
    '''
    unit = 20
    if overide:
        tt_acu = pickle.load(open("DATA/drop_out_loss_stats_0526.p", "rb"))
        print('finish load drop_out_plot_stats data from pickle!')
        return tt_acu
    else:
        # train & test accuracy
        tt_acu = []
        # final_data = contextualize(None, True)

        print('Begin training for drop_out_vs_loss!')
        for drop_out in reversed(range(1,unit+1)):
            print('epoch', unit - drop_out)

            drop_out = drop_out/unit
            print('drop out:',1-drop_out)
            print('len of train data:',len(train_data))
            print('len of cv data:', len(cv_data))
            train_loss_, cv_loss_, train_f1_score, cv_f1_score= train(train_data,cv_data, drop_out = drop_out)
            tt_acu.append([1-drop_out,train_loss_,cv_loss_,train_f1_score,cv_f1_score])
        print('result of cross validation:')
        print(tt_acu)
        pickle.dump(tt_acu, open("DATA/drop_out_loss_stats_0526.p", "wb+"))
        print('-success dump drop_out_plot_stats data-')
        return tt_acu

# ------------------------------
# plot loss of different L2 regulariziation term
# ------------------------------
def l2_vs_loss(overide=False):
    '''
    : input
    '''

    unit = 20

    if overide:
        tt_acu = pickle.load(open("DATA/l2_loss_stats_0526.p", "rb"))
        print('finish load l2_plot_stats data from pickle!')
        return tt_acu
    else:
        # train & test accuracy
        tt_acu = []
        # final_data = contextualize(None, True)

        print('Begin training for l2_vs_loss!')
        for l2_term in range(unit+1):
            print('epoch', l2_term)

            l2_term = l2_term/unit
            # l2_term=np.log(l2_term+1)
            print('l2_term:',l2_term)

            train_loss_,cv_loss_,train_f1_score,cv_f1_score = train(train_data,cv_data, beta_l2 = l2_term)
            tt_acu.append([l2_term,train_loss_,cv_loss_,train_f1_score,cv_f1_score])
        print('result of cross validation:')
        print(tt_acu)
        pickle.dump(tt_acu, open("DATA/l2_loss_stats_0526.p", "wb+"))
        print('-success dump l2_plot_stats data-')
        return tt_acu

# ------------------------------
# plot loss of different L2 regulariziation term
# ------------------------------
def l1_vs_loss(overide=False):
    '''
    : input
    '''
    unit = 20
    if overide:
        tt_acu = pickle.load(open("DATA/l2_loss_stats_clustered.p", "rb"))
        print('finish load l2_plot_stats data from pickle!')
        return tt_acu
    else:
        # train & test accuracy
        tt_acu = []
        # final_data = contextualize(None, True)

        print('Begin training for l1_vs_loss!')
        for l1_term in range(1,unit+1):
            print('epoch', l1_term)

            l1_term = l1_term/unit
            # l1_term=np.log(l1_term)
            print('l1_term:',l1_term)

            train_loss_,cv_loss_,train_f1_score,cv_f1_score = train(train_data,cv_data, beta_l2 = l1_term,l1=True)
            tt_acu.append([l1_term,train_loss_,cv_loss_,train_f1_score,cv_f1_score])
        print('result of cross validation:')
        print(tt_acu)
        pickle.dump(tt_acu, open("DATA/l1_loss_stats_clustered.p", "wb+"))
        print('-success dump l1_plot_stats data-')
        return tt_acu

# ------------------------------
# plot
# ------------------------------
def plot(tt_acu,xlabel,plot_path='default_save.png'):
    '''
    :input data: a list of sublist, one sublist contains data through time of training or cross validation
    '''

    plt.xlabel(xlabel)

    df = pd.DataFrame.from_records(data = [(i[1],i[2]) for i in tt_acu],index =list(zip(*tt_acu))[0], columns=['train','cv'])
    plt.subplot(2, 1, 1)
    plt.plot(df)
    plt.ylabel('Loss')
    plt.legend(labels=['train','cv'])

    df2 = pd.DataFrame.from_records(data = [(i[3],i[4]) for i in tt_acu],index =list(zip(*tt_acu))[0], columns=['train','cv'])
    plt.subplot(2, 1, 2)
    plt.plot(df2)
    plt.ylabel('F1')
    plt.legend(labels=['train','cv'])

    plt.savefig(plot_path)
    plt.show()


if __name__ == '__main__':


    '''
    with original 12k data
    '''
    data_file = 'sms_clean_0524.txt'

    vocab_str, vocab_vec = loadZhW2v(None, True)
    s = np.vstack(vocab_vec)

    Gvar = np.var(s, 0)  # distributional parameter for Glove, for later generating random embedding for UNK
    Gmean = np.mean(s, 0)
    raw_data = read_data(None, True)

    final_data = contextualize(raw_data=None,overide=True)
    # plot datasize_vs_loss
    tt_acu = datasize_vs_loss()
    plot(tt_acu,"size of dataset","PLOT/clean_datasize_loss_0604.png")

    # # plot drop_out_vs_loss
    # train_data, cv_data, _ = split_data(final_data, 0)
    # tt_acu = drop_out_vs_loss()
    # plot(tt_acu,"Dropout","PLOT/dropout_loss_stats.png")

    # plot L2 regulariziation term _vs_loss
    train_data, cv_data, _ = split_data(final_data, 0)
    tt_acu = l2_vs_loss()
    plot(tt_acu,"L2 regularization term","PLOT/l2_loss_stats_0604.png")

    '''
    use clustered data
    '''
    # data_file = 'clustered_sms_message.txt'
    #
    # vocab_str, vocab_vec = loadZhW2v(None, True)
    # s = np.vstack(vocab_vec)
    #
    # Gvar = np.var(s, 0)  # distributional parameter for Glove, for later generating random embedding for UNK
    # Gmean = np.mean(s, 0)


    # raw_data = read_data(data_file)
    #
    # final_data = contextualize(raw_data,vocab_str, vocab_vec, Gmean, Gvar)
    # # plot datasize_vs_loss
    # tt_acu = datasize_vs_loss()
    # plot(tt_acu,"size of dataset","PLOT/datasize_loss_stats_clustered.png")
    #
    #
    # # plot L2 regulariziation term _vs_loss
    # train_data, cv_data, _ = split_data(final_data, 0)
    # tt_acu = l2_vs_loss()
    # plot(tt_acu,"L2 regularization term","PLOT/l2_loss_stats_clustered.png")
