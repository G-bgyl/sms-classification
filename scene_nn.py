# -*- coding: utf-8 -*-
import csv
import re
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
import tensorflow as tf
import datetime
import time
import usrCut

# distinguish sms type
target_scene={'S0001':0,'S0002':1,'S0003':2,'S0004':3,'S0005':4,'S0006':5,'S0007':6,'S0008':7}
maxSeqLength=75
# --------------------------
# load chinese word2vec
# --------------------------
def loadZhW2v(zh_word_vec_file=None,overide = False):
    """
    # input : word2vec_file
    # output :
    # a list of string of word segments
    # a list of vector(list) of word segments

    """
    if overide:
        vocab_str = pickle.load(open("DATA/vocab_str.p", "rb"))
        vocab_vec = pickle.load(open("DATA/vocab_vec.p", "rb"))
        print('finish load vocab_str & vocab_vec from pickle!')
        return vocab_str , vocab_vec
    else:
        loadw2v_start_time = time.time()
        with open (zh_word_vec_file,'r')as word2vec:
            lines = csv.reader(word2vec)
            vocab_str = []
            vocab_vec = []
            i=0
            for line in lines:
                i+=1
                # each line is a list with only one string.
                str_line = line[0]

                # process word vector that is not strictly a list

                if '\t' in str_line: # means this line contains the word segment string
                    one_seg = str_line.split('\t')
                    str_word_seg = one_seg[1]
                    vec_word_seg = one_seg[2]
                elif ']' in str_line: # last line of a word segment's vector
                    vec_word_seg += str_line
                    vocab_str.append(str_word_seg)
                    # turn string into list and then convert list of strings into list of float
                    vec_word_seg = list(map(float, vec_word_seg.strip('[]').split()))
                    #convert list into np array
                    vocab_vec.append(np.asarray(vec_word_seg))
                    str_word_seg=''
                    vec_word_seg=[]
                else:# not finish a word segment's vector
                    vec_word_seg += str_line
            # save it in pickle to re-use next time
            pickle.dump(vocab_str, open("DATA/vocab_str.p", "wb+"))
            pickle.dump(vocab_vec, open("DATA/vocab_vec.p", "wb+"))

            loadw2v_finish_time = time.time()
            spend_time = float((loadw2v_start_time - loadw2v_finish_time)/60)
            print("======== total spend for create vocab_str & vocab_vec:",spend_time,'minutes ========\n')

            return vocab_str, vocab_vec


# -------------
# Dealing with Unknown words in zh_w2v
# -------------

def generate_new(unk,vocab_str, vocab_vec, Gmean, Gvar):
    """
      Input: unknown word
      Output: generate a random word embedding from multi-nomial distribution and add to glove_wordmap
    """
    # global vocab_str,vocab_vec, Gmean, Gvar

    RS = np.random.RandomState()
    # append new word string into vocab_str
    vocab_str.append(unk)
    # create new word vector based on normal distribution into vocab_vec
    vocab_vec.append(RS.multivariate_normal(Gmean, np.diag(Gvar)))

    return vocab_vec[-1]


# ---------------------------------------------------------------
# read data set, return the string of content and its label
# ---------------------------------------------------------------
def read_data(file_name,name,overide = False):
    # input: csv file name string, 'sms_messages.csv', when overide, set as None
    # columns of input file: id,raw_content,scene_code
    # output:
    # the raw content as a list of string tokens.
    # the label of content whether it belongs to target_scene or not.

    if overide:
        raw_data = pickle.load(open(name, "rb"))
        print('finish load raw_data from pickle!')
        return raw_data
    else:
        read_data_start_time = time.time()
        raw_data = []
        with open(file_name) as sms_messages:
            reader = csv.reader(sms_messages,delimiter='\t')
            next(reader)
            for each in reader:
                string_content = each[1]
                labels = np.zeros((8,))
                labels[target_scene[each[2]]]=1
                raw_data.append((int(each[0]),string_content,labels))
        read_data_finish_time = time.time()
        spend_time = float((read_data_finish_time - read_data_start_time)/60)
        print("======== total spend for process raw_data:",spend_time,'minutes ========\n')
        pickle.dump(raw_data, open(name, "wb+"))
        return raw_data



# -----------------------
# remove punctuations and word segmentation
# -----------------------
def cleanSentences(sentence):
    s = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）：；《）【】《》“”()»〔〕-]+", "",
               sentence) # .decode("utf8")
    return usrCut.date_string_split(s)




# ----------------------------------
# turn sentence into list of index
# ----------------------------------
def sentence2sequence(sentence,vocab_str, vocab_vec, Gmean, Gvar):
    """
        - Turns an input paragraph into an (m,d) matrix,
            where n is the number of tokens in the sentence
            and d is the number of dimensions each word vector has.
          Input: sentence, string
          Output: list of index of each word segment in sentence
    """
    sent_raw_list = cleanSentences(sentence)
    sent_str_list=[]
    sent_idx_list = []

    for j in range(len(sent_raw_list)):
        if len(sent_idx_list) < maxSeqLength:
            word = sent_raw_list[j]
            try:
                sent_idx_list.append(vocab_vec[vocab_str.index(word)])
                sent_str_list.append(word)
            except:
                # Greedy search for word: if it did not find the whole word in zh_w2v, then truncate tail to find
                i = len(word)
                while len(word) > 0:
                    wd = word[:i]  # shallow copy
                    if wd in vocab_str:
                        # Make sure the length of th list is less than maxSeqLength
                        if len(sent_idx_list) < maxSeqLength:
                            sent_idx_list.append(vocab_vec[vocab_str.index(wd)])
                            sent_str_list.append(wd)
                            word = word[i:]  # set to whatever the rest of words is, eg. hallway --> hall, way, split to 2 words
                            i = len(word)
                            continue
                        else: break
                    else:
                        i = i - 1
                    if i == 0:
                        # Make sure the length of th list is less than maxSeqLength
                        if len(sent_idx_list) < maxSeqLength:
                            sent_idx_list.append(generate_new(word,vocab_str, vocab_vec, Gmean, Gvar))
                            sent_str_list.append(word)
                            break
                        else: break
        else: break
    # make sure the length of two return params are fixed.
    left_len = maxSeqLength-len(sent_idx_list)
    if left_len>0:
        sent_idx_list.extend(np.zeros((300,)) for i in range(left_len))
        sent_str_list.extend([''] * left_len)
    return sent_str_list,sent_idx_list

# -----------------------------
# read in and prepare data
# -----------------------------
def contextualize(raw_data,name,vocab_str=None, vocab_vec=None, Gmean=None, Gvar=None, overide = False):
    '''
    :param raw_data: output of read_data
    :param overide: use pickle or not
    :return: final data contains (labels,sent_str_list,sent_idx_list)
    '''
    contextualize_start_time = time.time()
    if overide:
        data = pickle.load(open(name, "rb"))
        print('finish load final_data from pickle!')
        return data
    else:
        final_data = []
        i=0
        for each in raw_data:
            id,string_content, labels = each
            sent_str_list, sent_idx_list = sentence2sequence(string_content,vocab_str, vocab_vec, Gmean, Gvar)
            final_data.append((int(id), labels,sent_str_list,sent_idx_list))
            if i%500 ==0:
                print(i,len(sent_str_list),len(sent_idx_list[0]),'process sms message in sentence2sequence!')
            i+=1
        pickle.dump(final_data, open(name, "wb+"))



        contextualize_finish_time = time.time()
        spend_time = float((contextualize_finish_time - contextualize_start_time)/60)
        print("======== total spend for contextualize final data:",spend_time,'minutes ========\n')

        return final_data


# ----------------------------------------
# split data set into train, dev, test
# ----------------------------------------
def split_data(final_data,train_size = 0, overide = False):
    '''

    :param final_data: (id,raw_content, scene_category), None when overide is True
    :param train_size: if = 0 , means use full length of training. if = 0.9, means only use 10% of training set.
    :return: seperate data into train, dev, test
    '''

    if overide:
        if overide =='train':
            #  final data would be cluastered data,and what in pickle would be whole clean data,must with id.

            train_data=[]

            final_whole_data_ = pickle.load(open("DATA/final_data_w_id0530.p", "rb"))

            train_dev, test_data = train_test_split(final_whole_data_, test_size=0.2, random_state=42)

            train_data_all, cv_data = train_test_split(train_dev, test_size=0.25, random_state=42)
            clustered_train_id = list(zip(*final_data))[0]
            for each in train_data_all:
                if each[0] in clustered_train_id:
                    train_data.append(each)
            train_data, _ = train_test_split(train_data, test_size=train_size, random_state=42)
            print('train mode -- len of train data:',len(train_data))
            pickle.dump(train_data, open("DATA/train_data_clus_w_id.p", "wb+"))
            pickle.dump(cv_data, open("DATA/cv_data_w_id.p", "wb+"))
            pickle.dump(test_data, open("DATA/test_data_w_id.p", "wb+"))

            print('train mode -- finish dump split data from pickle!')
            return train_data, cv_data, test_data
        else:
            train_data = pickle.load(open("DATA/train_data.p", "rb"))
            cv_data = pickle.load(open("DATA/cv_data.p", "rb"))
            test_data = pickle.load(open("DATA/test_data.p", "rb"))
            print('finish load split data from pickle!')
            return train_data, cv_data, test_data
    else:
        train_dev, test_data = train_test_split(final_data, test_size=0.2,random_state=42)
        train_data , cv_data = train_test_split(train_dev, test_size=0.25,random_state=42)
        train_data, _ = train_test_split(train_data, test_size=train_size,random_state=42)
        pickle.dump(train_data, open("DATA/train_data.p", "wb+"))
        pickle.dump(cv_data, open("DATA/cv_data.p", "wb+"))
        pickle.dump(test_data, open("DATA/test_data.p", "wb+"))
        print('finish split data!')
        return train_data, cv_data, test_data





# ----------------
#  Prepare batch
# ----------------
def prepare_batch(batch_data):
    '''

    :param batch_data: subset of final_data
    :return: labels, sent_str_list, sent_idx_list
    '''
    labels, sent_str_list, sent_idx_list = zip(*batch_data)
    return zip(*batch_data)


batchSize = 200
lstmUnits = 64
numClasses = 8
iterations = 12000
numDimensions = 300

# ----------------------------------------------------------------
#  Specifically calculate stats for one type of sms_messages
# ----------------------------------------------------------------
def print_stats(nextBatchLabels,accuracy_num, correctPred_list, prediction_list,scene_code = 'S0002'):
    # calculate f1 score
    tp,fp,fn,tn=0,0,0,0

    # for scene_code in target_scene:
    # type_to_eval=1 when test S0002
    type_to_eval = target_scene[scene_code]
    for i in range(len(correctPred_list)):
        pred = list(prediction_list[i])
        accu = list(nextBatchLabels[i])
        if pred.index(max(pred))==type_to_eval:

            if accu.index(1)==type_to_eval:
                tp+=1
            else:
                fn+=1
        else:
            if accu.index(1)==type_to_eval:
                fp+=1
            else:
                tn+=1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy_ = (tp + tn) / (tp +fp + tn + fn)
    f1_score = 2/ (1/precision + 1/recall)
    print('............ stats for %s ............'%(scene_code))
    print('tp', tp)
    print('fp', fp)
    print('fn', fn)
    print('tn', tn)
    print('Precision for %s:'%(scene_code),precision)
    print('Recall for %s:'%(scene_code), recall)
    print('Accuracy for %s:'%(scene_code), accuracy_)
    print('F1 for %s:'%(scene_code), f1_score)
    return precision,recall,accuracy_,f1_score



def train(train_data,cv_data,drop_out=0.75,beta_l2 = 0.01,l1=False,overide = False):
    train_start_time = time.time()

    sess = tf.InteractiveSession()
    # saver = tf.train.Saver()
    # saver.restore(sess, tf.train.latest_checkpoint('models'))


    '''
    # set tensorflow parameters.
    '''
    tf.reset_default_graph()
    labels = tf.placeholder(tf.float32, [None, numClasses])  # batchSize
    # maybe is a batch size data
    batch_vec_data = tf.placeholder(tf.float32, [None, maxSeqLength, numDimensions], "context")  # batchSize

    BETA = beta_l2
    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=drop_out)
    value, _ = tf.nn.dynamic_rnn(lstmCell, batch_vec_data, dtype=tf.float32)
    weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]),name='WEIGHTS')
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    prediction = (tf.matmul(last, weight) + bias)

    correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))


    # l1 regularization :
    l1_regularizer = tf.contrib.layers.l1_regularizer(
        scale=BETA, scope=None
    )
    weights = tf.trainable_variables()
    regularization_penalty_l1 = tf.contrib.layers.apply_regularization(l1_regularizer,weights) #+ tf.contrib.layers.apply_regularization(l1_regularizer, bias)

    # l2 regularization
    regularization_penalty_l2 = BETA * tf.nn.l2_loss(weight) + BETA * tf.nn.l2_loss(bias)
    if l1 ==True:
        regularization_penalty=regularization_penalty_l1
    else:
        regularization_penalty=regularization_penalty_l2
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels) +regularization_penalty)
    optimizer = tf.train.AdamOptimizer(beta1=0.9,beta2=0.999).minimize(loss)



    '''
    # initial session
    '''
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    tf.summary.scalar('Loss', loss)
    tf.summary.scalar('Accuracy', accuracy)
    merged = tf.summary.merge_all()
    logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    writer = tf.summary.FileWriter(logdir, sess.graph)

    convergence = 0

    max_acc= 50
    record_mx=list(np.random.random_sample((50,)))
    '''
    # begin training
    '''
    for i in range(iterations):


        if convergence > 300:
            print('\nIteration %s Successfully converge!'%(i))
            break
        batch = np.random.randint(len(train_data), size=batchSize)
        batch_data = [train_data[k] for k in batch]
        # zip(*batch_data) returns labels, sent_str_list, sent_idx_list
        id, nextBatchLabels, _, nextBatchVec = zip(*batch_data)
        sess.run(optimizer, {batch_vec_data: nextBatchVec, labels: nextBatchLabels})
        feed_dict = {batch_vec_data: nextBatchVec, labels: nextBatchLabels}

        accuracy_num, correctPred_list = sess.run([accuracy,correctPred],
                                                          feed_dict=feed_dict)
        train_accu = (accuracy_num * 100)
        if (i+1) % 50 ==0:
            print("Accuracy for batch %s : %s %% "%((i+1),train_accu))

        # print out different result with label
        # for i in range(len(correctPred_list)):
        #     if correctPred_list[i]==False:
        #         wrong_sms = batch_data[i]
        #         print(wrong_sms[0], ''.join(wrong_sms[1]))
        # for i in range(batchSize):

        # Write summary to Tensorboard
        if (i % 20 == 0):
            summary = sess.run(merged, {batch_vec_data: nextBatchVec, labels: nextBatchLabels})
            writer.add_summary(summary, i)

    #     # Save the network every 10,000 training iterations
    #     if ((i+1) % 200 == 0 and i != 0):
    #         #  datetime.datetime.now().strftime("%Y%m%d-%H:%M")
    #         save_path = saver.save(sess, "models/"+"_pretrained_lstm.ckpt", global_step=i)
    # # datetime.datetime.now().strftime("%Y%m%d-%H:%M")+
    #         print("saved to %s" % save_path)
        # copy the old max number
        max_old=  [max(record_mx)].copy()[0]
        record_mx.pop(0)
        record_mx.append(train_accu)
        if max_old == max(record_mx):
            # record_mx[record_mx.index(min(record_mx))]=train_accu
            convergence +=1
        else:
            convergence = 0

        # help made judgment of converging.
        # old algorithm
        # if (train_accu-max_acc)/max_acc<0.00005:
        #
        #     convergence += 1
        #
        # else:
        #     max_acc = train_accu
        #     convergence = 0


    '''
    # calculate train data accuracy
    '''
    # because of split data in train mode, train data is processed with id,but cv_data(below) is not.

    id, nextBatchLabels, _, nextBatchVec = zip(*train_data)
    feed_dict = {batch_vec_data: nextBatchVec, labels: nextBatchLabels}
    accuracy_num, correctPred_list, prediction_list,loss_train = sess.run([accuracy, correctPred,prediction,loss],
                                          feed_dict=feed_dict)

    print('\nCalculate train data accuracy!')
    # these 4 stats are for 'S0002' by default
    train_precision,train_recall,train_accuracy_,train_f1_score = print_stats(nextBatchLabels,accuracy_num, correctPred_list, prediction_list)
    writer.close()


    '''
    Test accuracy
    '''
    # because of split data in train mode, train data(up there) is with id,but cv_data is not.
    # zip(*batch_data) returns labels, sent_str_list, sent_idx_list
    id, nextBatchLabels, nextBatchStr, nextBatchVec = zip(*cv_data)
    feed_dict =  {batch_vec_data: nextBatchVec, labels: nextBatchLabels}
    accuracy_num, correctPred_list, prediction_list,loss_cv = sess.run([accuracy, correctPred,prediction,loss],
                                          feed_dict=feed_dict)
    print('\nCalculate cross validation data accuracy!')
    # these 4 stats are for 'S0002' by default
    cv_precision,cv_recall,cv_accuracy_,cv_f1_score = print_stats(nextBatchLabels,accuracy_num, correctPred_list, prediction_list)



    # count number of different results
    j = 0
    for i in range(len(correctPred_list)):
        if correctPred_list[i] == False:
            j += 1

            # # print out different result with label
            # wrong_sms = nextBatchStr[i]
            # pred = list(prediction_list[i])
            # accu = list(nextBatchLabels[i])
            # print('%s\t%s\t%s'%(''.join(wrong_sms),accu.index(1),pred.index(max(pred))))
    test_accu = accuracy_num * 100
    # Accuracy over 8 types.
    print("\nAccuracy for test data:", test_accu)
    print('length of test data:',len(nextBatchVec))
    print('wrong of test data:', j)

    print('\nloss for train and cv:%s %s\n'%(loss_train,loss_cv))

    finish_time = time.time()
    spend_time = float((finish_time - train_start_time)/60)
    print("======== total spend for train:",spend_time,'minutes ========\n')

    sess.close()
    return loss_train,loss_cv

if __name__=='__main__':

    data_file = 'sms_clean_0524.txt'
    zh_word_vec_file = 'zh_w2v.tsv'


    # # TODO: need to rerun to update the type of elements from list into ndarray
    # vocab_str, vocab_vec = loadZhW2v(None, True)
    # s = np.vstack(vocab_vec)
    #
    # Gvar = np.var(s, 0)  # distributional parameter for Glove, for later generating random embedding for UNK
    # Gmean = np.mean(s, 0)
    #
    # raw_data = read_data(None, True)
    # # TODO: uncomemnt when need to rerun
    final_data = contextualize(raw_data=None, overide=True)
    train_data, cv_data, test_data = split_data(final_data,0)

    train(None, None, overide = True)
