import csv
import os
from imutils import paths
import random

root = './dataset/'
folder_name_list = os.listdir(root)
folder_name_list.sort() 
print('number of classes (or number of product ID): ', len(folder_name_list))

with open('data_train.csv', 'a') as csvfile: 
    writer = csv.writer(csvfile)
    for i, folder_name in enumerate(folder_name_list):
        folder_path = root + folder_name
        img_name_list = os.listdir(folder_path)

        for img_name in img_name_list:
            img_path = os.path.join(root, folder_name, img_name)
            label = i
            writer.writerow([img_path, label])

