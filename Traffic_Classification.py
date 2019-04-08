#!/usr/bin/env python
# coding: utf-8

# In[9]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from skimage.feature import hog
from sklearn.svm import SVC 
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report


# In[11]:


data_path = "/scratch/cmpe255-sp19/data/pr2/traffic"
color_space = 'YUV' 
orient = 15  
pix_per_cell = 10 
cell_per_block = 1 
hog_channel = "ALL" 
spatial_size = (40, 40) 
hist_bins = 40    
spatial_feat = True
hist_feat = True 
hog_feat = True 

X_train = []
y_train = []
X_test = []


# In[12]:


def loadLabels(fileName):
    labels = []
    with open(fileName) as my_file:
        for line in my_file:
            labels.append(line)
    return labels


# In[13]:


def loadImages(path):
    image_files = sorted([os.path.join(path, file) for file in os.listdir(path)])
    return image_files


# In[14]:


def image_processing(data):
    img = [cv2.imread(i,1) for i in data]
    w = 40
    h = 40
    dimension = (w, h)
    result = []
    for i in range(len(img)):
        res = cv2.resize(img[i], dimension, interpolation=cv2.INTER_CUBIC)
        result.append(res)
    return result


# In[22]:


def hog_features_descriptor(img, orient, pix_per_cell, cell_per_block):
    features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False,
                       visualise=False, feature_vector=True)
    return features


# In[23]:


def extract_features(image):
    yuv_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    car_ch1 = yuv_image[:,:,0]
    hog_features = hog_features_descriptor(car_ch1, orient, pix_per_cell, cell_per_block)
    return hog_features


# In[24]:


def extract_features_of_dataset(dataset):
    hog_features = []
    for image in dataset:
        hog_features.append(extract_features(image))
    return hog_features


# In[19]:


train_images = loadImages(data_path+"/train")
test_images = loadImages(data_path+"/test")
train_labels = loadLabels(data_path+"/train.labels")


# In[20]:


train_set_images = image_processing(train_images)
test_set_images = image_processing(test_images)


# In[22]:


# from sklearn.decomposition import PCA
# pca = PCA(n_components=21)
# principalComponents_train = pca.fit_transform(train_set_images)


# In[23]:


# pca = PCA().fit(train_set_images)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance')


# In[24]:


# pca = PCA(n_components=21)
# principalComponents_test = pca.fit_transform(test_set_images)


# In[21]:


# pca = PCA().fit(test_set_images)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance')


# In[25]:


train_set_hog_features = extract_features_of_dataset(train_set_images)
test_set_hog_features = extract_features_of_dataset(test_set_images)
X_train = np.array(train_set_hog_features)
y_train = np.array(train_labels)
X_test = np.array(test_set_hog_features)


# In[26]:


svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train, y_train)


# In[27]:


y_pred = svclassifier.predict(X_test)
# f= open("prd.dat","w+")
# for i in y_pred:
#     f.write(str(i))


# In[44]:


y_pred


# In[45]:


predictions_file = open("prp.dat","w")
for label in y_pred:
    predictions_file.write(str(int(label)))
    predictions_file.write("\n")
predictions_file.close()


# In[58]:


# from sklearn.ensemble import RandomForestClassifier
# rndm_clf = RandomForestClassifier(class_weight=None, criterion='gini', max_features='auto', min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, n_estimators=300,random_state=0, verbose=0)
# rndm_clf = rndm_clf.fit(X_train, y_train)
# y_pred_rand = rndm_clf.predict(X_test)
# #print(classification_report(test_labels, y_pred_rand)) 


# In[59]:


# from sklearn.neural_network import MLPClassifier
# clf = MLPClassifier(hidden_layer_sizes=(100, 4), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=400, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
# clf.fit(principalComponents, train_labels)

# y_pred_ann = clf.predict(principalComponents_test)
# print(classification_report(test_labels, y_pred_ann))


# In[60]:


# from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
# clf = clf.fit(principalComponents, train_labels)
# y_pred_svd = clf.predict(principalComponents_test)
# print(classification_report(test_labels, y_pred_svd))


# In[61]:


# import xgboost as xgb
# xg_reg = xgb.XGBClassifier(objective ='multi:softmax' ,learning_rate = 0.63, 
#                           max_depth = 6, n_estimators = 400, num_class = 12)
# xg_reg.fit(principalComponents,train_labels)
# preds = xg_reg.predict(principalComponents_test)
# print(classification_report(test_labels, preds))


# ## Tried On Traffic Small Dataset in local

# In[62]:


# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPooling2D
# from keras.utils import to_categorical
# from keras.preprocessing import image
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from keras.utils import to_categorical
# from tqdm import tqdm


# In[63]:


# train_labels_ = np.loadtxt('/Users/Rutul Thakkar/Desktop/Sjsu Jupyter/CMPE255/Program2/traffic/traffic-small/train.labels')
# train_labels = pd.DataFrame(train_labels_)


# In[64]:


# import glob   
# train_path  = '/Users/Rutul Thakkar/Desktop/Sjsu Jupyter/CMPE255/Program2/traffic/traffic-small/train/*.jpg'
# train_files=glob.glob(train_path)

# train_image = []
# for train_file in train_files:
#     img = image.load_img(train_file, target_size=(28,28,1), grayscale=True)
#     img = image.img_to_array(img)
#     img = img/255
#     train_image.append(img)
# X = np.array(train_image)


# In[65]:


# train_labels = to_categorical(train_labels)


# In[66]:


# test_path = '/Users/Rutul Thakkar/Desktop/Sjsu Jupyter/CMPE255/Program2/traffic/traffic-small/test/*.jpg'
# test_files=glob.glob(path)

# test_image = []
# for test_file in test_files:
#     img = image.load_img(test_file, target_size=(28,28,1), grayscale=True)
#     img = image.img_to_array(img)
#     img = img/255
#     test_image.append(img)
# test = np.array(test_image)


# In[67]:


# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(15, activation='softmax'))
# model.summary()


# In[68]:


# model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
# model.fit(X, train_labels,batch_size= 100 ,epochs=10)


# In[69]:


# prediction = model.predict_classes(test)


# In[70]:


# prediction_labels = to_categorical(prediction)


# In[71]:


# test_labels_ = np.loadtxt('/Users/Rutul Thakkar/Desktop/Sjsu Jupyter/CMPE255/Program2/traffic/traffic-small/test.labels')
# test_labels_ = pd.DataFrame(test_labels_)


# In[72]:


# test_labels = to_categorical(test_labels_)


# In[74]:


# from sklearn.metrics import f1_score
# f1 = f1_score(test_labels, prediction_labels, average='weighted') 


# In[ ]:





# In[ ]:




