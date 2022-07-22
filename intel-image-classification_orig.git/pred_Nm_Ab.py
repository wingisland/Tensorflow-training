import tensorflow as tf
from keras.models import load_model
from tensorflow.keras import layers as layers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.keras import Sequential
from sklearn.utils import shuffle
import tensorflow.keras.optimizers as Optimizer
import matplotlib.pyplot as plot
import numpy as np
import os , os.path
from tensorflow.keras.models import Sequential, save_model, load_model
from keras.preprocessing import image
from random import randint
from matplotlib.ticker import MultipleLocator
import matplotlib.gridspec as gridspec
import time
import math
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import shutil
import csv
import re
from tkinter import _flatten
import ast
from main import train

#Contro pannel
########################################################
saved_model_to_pred  = './saved_model/2022-06-17 13:42:04_important'
train_new_model = 0
filepath = './saved_model'         #model path
timenow=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
true_status         = 'Abnormal'

########################################################
def getImages(dataset_dir, img_size):

    dataset_array  = []
    dataset_labels = []
    dataset_directory = []
    class_counter = 0
    class_names = os.listdir(dataset_dir)
    sort_array     = []
    sorted_dataset_directory  = []
    sorted_class_names        = []
    sorted_dataset_labels     = []
    sorted_dataset_array      = []


    for current_class_name in class_names:
        # Get class directory
        class_dir = os.path.join(dataset_dir, current_class_name)

        # Keep track of the class that is being extracted
        images_in_class = os.listdir(class_dir)
        print("Class index", class_counter, ", ", current_class_name, ":", len(images_in_class))
        for image_file in images_in_class:
            if image_file.endswith(".png"):
                image_file_dir = os.path.join(class_dir, image_file)
                img = tf.keras.preprocessing.image.load_img(image_file_dir, target_size=(img_size, img_size))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = img_array / 255.0
                dataset_array.append(img_array)
                dataset_labels.append(class_counter)
                dataset_directory.append(image_file)
                sort_array.append([image_file, class_names, class_counter, img_array])
        # Increase the counter when we're done with a class
        class_counter += 1

    # Shuffle both lists the same way
    dataset_array, dataset_labels, dataset_directory = shuffle(dataset_array, dataset_labels, dataset_directory, random_state=817328462)
    # Transform to a numpy array
    dataset_array  = np.array(dataset_array)
    dataset_labels = np.array(dataset_labels)
    dataset_directory = np.array(dataset_directory)


    '''
    one_ful_array  = []
    index  = []
    index2 = []
    index3 = []
    index4 = []
    sort_array.sort()

    for i in sort_array :
        one_ful_array  = sort_array[i]
        index = one_ful_array[0]
        sorted_dataset_directory.append(index2)
        index2 = one_ful_array[1]
        sorted_class_names.append(index2)
        index3 = one_ful_array[2]
        sorted_dataset_labels.append(index2)
        index4 = one_ful_array[3]
        sorted_dataset_array.append(index2)
    '''

    return dataset_array, dataset_labels, class_names, dataset_directory



def start_prediction():
    # get prdeict image
    pred_ds, pred_classes, pred_class_names, pred_directory = getImages('./data/seg_pred/',150)
    DIR = './data/seg_pred/seg_pred'
    Predition_data_count=len([chunk for chunk in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, chunk))])
    print ("Predition data count",Predition_data_count)
    
    idle_data_count     = 0
    working_data_count  = 0
    invalid_value_count = 0
    pred_list           = []
    invalid_cnt         = 0
    white_list          = []
    pred_status         = ''

    #workfine_time_list = list(range(workfine_time[0], workfine_time[1]))
    #abnormal_time_list = list(range(abnormal_time[0], abnormal_time[1]))
    
    #print(list(workfine_time))
    model = load_model(saved_model_to_pred)
    model.summary()
    exit()
    for i in range(Predition_data_count):
        pred_img = np.array([pred_ds[i]])
        pred_prob = model.predict(pred_img)
        #ful_sort_path =f'{sort_path}/switch/{pred_directory[i]}'
        #print(pred_prob) 
        processed_pred_prob = pred_prob[0]
        first_index  = processed_pred_prob[0]
        #second_index = processed_pred_prob[1]
        #third_index  = processed_pred_prob[2]
        #print(processed_pred_prob)
        #print(pred_directory[i])
        directory_call    = str(pred_directory[i])
        directory_call    = directory_call.split(',')
        #print(white_list_string)
        with open('white_list.csv', 'r', newline='') as csv_read :
             reader = csv.reader(csv_read)
             for row in reader:
                 white_list.extend(row)
    
             if directory_call[0] not in white_list:
                with open('white_list.csv', 'a', newline='') as csv_write:
                     writer = csv.writer(csv_write)
                     writer.writerow(directory_call)
    
        if  first_index  > 0.5:
            pred_status = "Normal"
            #true_status = "none"
            idle_data_count  += 1
            pred_list.append("Normal")
            '''
           # shutil.copy(f'{DIR}/{pred_directory[i]}', ful_sort_path.replace("switch", "Idle"))            
            for i in range (0,len(abnormal_time_list)-1):
                ab = str(abnormal_time_list[i])
                if   f'_{ab}_' in directory_call[0] : 
                     invalid_cnt=invalid_cnt+1
                     true_status = "bad"
            
            if pred_status != true_status:
                 invalid_value_count += 1
            '''

            with open('pred_lookup_table.csv', 'a', newline='') as lookup_table:
                 writer = csv.writer(lookup_table)
                 writer.writerow([directory_call," " ,true_status ," " ,pred_status])
    
    
        else:
             pred_status = "Abnormal"
             #true_status = "none"
             working_data_count   += 1
             pred_list.append("Abnormal")
             '''
            # shutil.copy(f'{DIR}/{pred_directory[i]}', ful_sort_path.replace("switch", "Working"))
             for i in range (0,len(workfine_time_list)-1):
                 wft = str(workfine_time_list[i])
                 if   f'_{wft}_' in directory_call[0] : 
                      invalid_cnt=invalid_cnt+1
                      true_status = "good"
             if pred_status != true_status:
                  invalid_value_count += 1
             '''

             with open('pred_lookup_table.csv', 'a', newline='') as lookup_table:
                  writer = csv.writer(lookup_table)
                  writer.writerow([directory_call," " ,true_status ," " ,pred_status])
    '''
    #print("Idle size",idle_data_count)
    #print("Working size" ,working_data_count)
    #print("invalid value size" ,invalid_value_count)
    valid_value    = idle_data_count + working_data_count
    invalid_value  = invalid_value_count
    precision = (Predition_data_count-invalid_value)/Predition_data_count
    precision_round_number = round(precision, 2)
    #print("The precision is :", precision_round_number*100," % ")
    all_count      = [idle_data_count,working_data_count,invalid_value]
    labels = 'Normal', 'Abnormal' ,'invalid_value'
    with open('pred_lookup_table.csv', 'r', newline='') as csv_read :
         reader = csv.reader(csv_read)
         for row in reader:
             pred_list = [row[4] for row in reader]
            #print('B',flag_list)
    print(collections.Counter(pred_list))
    '''
    '''
    outer = gridspec.GridSpec(10, 10, wspace=1, hspace=1)
    labels_pos =[0,1,2]
    x = np.array(labels_pos)
    y = np.array(all_count)
    plot.bar(x,y)
    plot.title(f"The precision is {precision_round_number*100} % ")
    plot.xticks(labels_pos,labels)
    plot.show()
    '''
    '''
    pred_list   = []
    flag_list   = []
    
    with open('pred_lookup_table.csv', 'r', newline='') as csv_read :
         reader = csv.reader(csv_read)
         for row in reader:
             flag_list = [row[2] for row in reader]
            #print('A',pred_list)
    with open('pred_lookup_table.csv', 'r', newline='') as csv_read :
         reader = csv.reader(csv_read)
         for row in reader:
             pred_list = [row[4] for row in reader]
            #print('B',flag_list)


    y_true = flag_list
    y_pred = pred_list
    conmatrix = confusion_matrix(y_true, y_pred, labels=["Idle", "Working", "Invalid_value"])
    #print(conmatrix)
    df_cm = pd.DataFrame(conmatrix, index = [i for i in labels],columns = [i for i in labels])
    plot.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=True)
    plot.xlabel("Predicted Class")
    plot.ylabel("True Class")
    plot.show()
    '''
    
    
def self_training():
    print('start_self_training')
    
    
    
    
if __name__ == "__main__":
    if train_new_model == 1:
       train_ds, train_classes, class_names, pred_directory= getImages('./data/seg_train/seg_train/', 150)
       train(train_ds, train_classes, class_names)

    else:
       #model = load_model(saved_model_to_pred)
       #model.summary()
       start_prediction()

