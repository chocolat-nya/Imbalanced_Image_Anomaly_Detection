import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from PIL import Image
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Activation, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
import time
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from efficientnet import *

epochs=500
training = True
model_number = 3

if model_number == 0:
    image_size = 224
elif model_number == 1:
    image_size = 240
elif model_number == 2:
    image_size = 260
elif model_number == 3:
    image_size = 300
elif model_number == 4:
    image_size = 380
elif model_number == 5:
    image_size = 456
elif model_number == 6:
    image_size = 528
elif model_number == 7:
    image_size = 600
else:
    print('set model_number 0 to 7')

ok_test_rate = 0.2
ng_test_rate = 0.2

class data:
    def resize(self, x, to_color=False):
        result = []

        for i in range(len(x)):
            if to_color:
                img = cv2.cvtColor(x[i], cv2.COLOR_GRAY2RGB)
                img = cv2.resize(img,dsize=(image_size,image_size))
            else:
                img = cv2.resize(x[i],dsize=(image_size,image_size))
            result.append(img)

        return np.array(result)

    def choose_data(self, x, y, ok_label, normal_id, anomaly_id):
        x_normal, x_anomaly = [], []
        x_ref, y_ref = [], []
        j = 0
        for i in range(len(y)):
            if y[i] == normal_id:
                x_normal.append(x[i].reshape((x.shape[1:])))
            elif y[i] == anomaly_id:
                x_anomaly.append(x[i].reshape((x.shape[1:])))
                if j < len(ok_label):
                   x_ref.append(x[i].reshape((x.shape[1:])))
                   y_ref.append(y[i])
                   j += 1

        return np.array(x_normal), np.array(x_anomaly), np.array(x_ref), y_ref

    def get_data(self):
        oks = glob.glob('images/ok/*')
        ngs = glob.glob('images/ng/*')
        ok_data = np.zeros((len(oks),image_size,image_size,3),dtype=np.float32)
        ok_label = np.zeros(len(oks))
        ng_data = np.zeros((len(ngs),image_size,image_size,3),dtype=np.float32)
        ng_label = np.zeros(len(ngs))
        normal_path = []
        anomaly_path = []

        for i in range(len(oks)):
           print(i+1,'/',len(oks))
           image = read_and_preprocess_img(oks[i],size=(image_size,image_size))
           ok_data[i] = image
           ok_label[i] = 0
           normal_path.append(oks[i])
        for i in range(len(ngs)):
           print(i+1,'/',len(ngs))
           image = read_and_preprocess_img(ngs[i],size=(image_size,image_size))
           ng_data[i] = image
           ng_label[i] = 1
           anomaly_path.append(ngs[i])

        ok_data_train, ok_data_test, ok_label_train, ok_label_test, normal_path_train, normal_path_test = train_test_split(ok_data, ok_label, normal_path, test_size = ok_test_rate, random_state=0)
        ng_data_train, ng_data_test, ng_label_train, ng_label_test, anomaly_path_train, anomaly_path_test = train_test_split(ng_data, ng_label, anomaly_path, test_size = ng_test_rate, random_state=0)

        x_train = np.concatenate((ok_data_train, ng_data_train),axis=0)
        x_test = np.concatenate((ok_data_test, ng_data_test),axis=0)
        y_train = np.concatenate((ok_label_train, ng_label_train),axis=0)
        y_test = np.concatenate((ok_label_test, ng_label_test),axis=0)

        x_train_normal, _, x_ref, y_ref = self.choose_data(x_train, y_train, ok_label_train, 0, 1)
        y_ref = to_categorical(y_ref)

        x_test_normal, x_test_anomaly, _, _ = self.choose_data(x_test, y_test, ok_label_test, 0, 1)

        x_train_normal = self.resize(x_train_normal)
        x_ref = self.resize(x_ref)
        x_test_normal = self.resize(x_test_normal)
        x_test_anomaly = self.resize(x_test_anomaly)

        return x_train_normal, x_ref, y_ref, x_test_normal, x_test_anomaly, normal_path_test, anomaly_path_test

class Arcfacelayer(Layer):
    def __init__(self, output_dim, s=30, m=0.50, easy_margin=False):
        self.output_dim = output_dim
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        super(Arcfacelayer, self).__init__()

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[0][1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(Arcfacelayer, self).build(input_shape)
 
    def call(self, x):
        y = x[1]
        x_normalize = tf.math.l2_normalize(x[0])
        k_normalize = tf.math.l2_normalize(self.kernel)

        cos_m = K.cos(self.m)
        sin_m = K.sin(self.m)
        th = K.cos(np.pi - self.m)
        mm = K.sin(np.pi - self.m) * self.m

        cosine = K.dot(x_normalize, k_normalize)
        sine = K.sqrt(1.0 - K.square(cosine))

        phi = cosine * cos_m - sine * sin_m

        if self.easy_margin:
            phi = tf.where(cosine > 0, phi, cosine) 

        else:
            phi = tf.where(cosine > th, phi, cosine - mm) 

        output = (y * phi) + ((1.0 - y) * cosine) 
        output *= self.s

        return output

    def compute_output_shape(self, input_shape):

        return (input_shape[0][0], self.output_dim)

def train_arcface(x, y, classes):
    print("ArcFace training...")

    model_name = globals()['EfficientNetB'+str(model_number)]
    base_model = model_name(weights='noisy-student',input_shape=x.shape[1:], include_top=False)

    c = base_model.output
    yinput = Input(shape=(classes,))
    hidden = GlobalAveragePooling2D()(c) 
    c = Arcfacelayer(classes, 30, 0.05)([hidden,yinput])
    prediction = Activation('softmax')(c)
    model = Model(inputs=[base_model.input, yinput], outputs=prediction)
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0001, amsgrad=True),
                  metrics=['accuracy'])

    es_cb = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=10, verbose=1, mode='auto')

    hist = model.fit([x, y], y, batch_size=4, epochs=epochs, verbose = 1, callbacks=[es_cb])

    return model

def no_train_arcface(x, y, classes):
    print("ArcFace training...")
    base_model=MobileNetV2(input_shape=x.shape[1:],alpha=0.5,
                           weights='imagenet',
                           include_top=False)

    c = base_model.output
    yinput = Input(shape=(classes,))
    hidden = GlobalAveragePooling2D()(c) 
    c = Arcfacelayer(classes, 30, 0.05)([hidden,yinput])
    prediction = Activation('softmax')(c)
    model = Model(inputs=[base_model.input, yinput], outputs=prediction)

#    model.compile(loss='categorical_crossentropy',
#                  optimizer=Adam(lr=0.0001, amsgrad=True),
#                  metrics=['accuracy'])

#    es_cb = keras.callbacks.EarlyStopping(monitor='loss', patience=50, verbose=0, mode='auto')

#    hist = model.fit([x, y], y, batch_size=4, epochs=epochs, verbose = 1, callbacks=[es_cb])

    return model

def get_score_arc(model, train, test):
    model = Model(model.get_layer(index=0).input, model.get_layer(index=-4).output)
    hold_vector = model.predict(train)
    predict_vector = model.predict(test)

    score = []

    for i in range(len(predict_vector)):
        cos_similarity = cosine_similarity(predict_vector[i], hold_vector)
        score.append(np.max(cos_similarity))
    return np.array(score)

def cosine_similarity(x1, x2): 
    if x1.ndim == 1:
        x1 = x1[np.newaxis]
    if x2.ndim == 1:
        x2 = x2[np.newaxis]
    x1_norm = np.linalg.norm(x1, axis=1)
    x2_norm = np.linalg.norm(x2, axis=1)
    cosine_sim = np.dot(x1, x2.T)/(x1_norm*x2_norm+1e-10)
    return cosine_sim

def softmax(x):
    f = np.exp(x)/np.sum(np.exp(x), axis = 0, keepdims = True)
    return f

def ScoreCam(model, hold_vector, img_array, layer_name, max_N=-1):

    cls = np.argmax(model.predict([img_array,hold_vector]))
    act_map_array = Model(inputs=model.inputs, outputs=model.get_layer(index=layer_name).output).predict([img_array,hold_vector])

    input_shape = model.layers[0].output_shape[0][1:]  # get input shape
    # 1. upsampled to original input size
    act_map_resized_list = [cv2.resize(act_map_array[0,:,:,k], (input_shape[0],input_shape[1]), interpolation=cv2.INTER_LINEAR) for k in range(act_map_array.shape[3])]
    # 2. normalize the raw activation value in each activation map into [0, 1]
    act_map_normalized_list = []
    for act_map_resized in act_map_resized_list:
        if np.max(act_map_resized) - np.min(act_map_resized) != 0:
            act_map_normalized = act_map_resized / (np.max(act_map_resized) - np.min(act_map_resized))
        else:
            act_map_normalized = act_map_resized
        act_map_normalized_list.append(act_map_normalized)
    # 3. project highlighted area in the activation map to original input space by multiplying the normalized activation map
    masked_input_list = []
    for act_map_normalized in act_map_normalized_list:
        masked_input = np.copy(img_array)
        for k in range(3):
            masked_input[0,:,:,k] *= act_map_normalized
        masked_input_list.append(masked_input)
    masked_input_array = np.concatenate(masked_input_list, axis=0)
    # 4. feed masked inputs into CNN model and softmax
    hold_vector2 = np.zeros((masked_input_array.shape[0], 2))
    hold_vector2[:,1] = 1
    pred_from_masked_input_array = softmax(model.predict([masked_input_array, hold_vector2]))
    # 5. define weight as the score of target class
    weights = pred_from_masked_input_array[:,cls]
    # 6. get final class discriminative localization map as linear weighted combination of all activation maps
    cam = np.dot(act_map_array[0,:,:,:], weights)
    cam = np.maximum(0, cam)  # Passing through ReLU
    cam = cam - np.min(cam)
    cam /= np.max(cam)  # scale 0 to 1.0
    
    return cam

def superimpose(original_img_path, cam, emphasize=False):
    
    img_bgr = cv2.imread(original_img_path)

    heatmap = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))
    if emphasize:
        heatmap = sigmoid(heatmap, 50, 0.5, 1)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    hif = .8
    superimposed_img = heatmap * hif + img_bgr
    superimposed_img = np.minimum(superimposed_img, 255.0).astype(np.uint8)  # scale 0 to 255  
 
    return superimposed_img

def sigmoid(x, a, b, c):
    return c / (1 + np.exp(-a * (x-b)))

def read_and_preprocess_img(path, size=(image_size,image_size)):
    img = load_img(path, target_size=size)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

DATA = data()

x_train_normal, x_ref, y_ref, x_test_normal, x_test_anomaly, normal_path_test, anomaly_path_test = DATA.get_data()

normal_label = np.zeros((len(x_train_normal), 2))
normal_label[:,0] = 1
anomaly_label = np.zeros((len(x_train_normal), 2))
anomaly_label[:,1] = 1

if training:
    model = train_arcface(np.vstack((x_train_normal, x_ref)), np.vstack((normal_label, y_ref)), y_ref.shape[1])
    model.save_weights('save.hdf5')
else:
    model = no_train_arcface(np.vstack((x_train_normal, x_ref)), np.vstack((normal_label, y_ref)), y_ref.shape[1])

model.load_weights('save.hdf5')
#model.summary()

Z1_arc = -get_score_arc(model, x_train_normal, x_test_normal)
Z2_arc = -get_score_arc(model, x_train_normal, x_test_anomaly)

result = np.empty((len(x_test_normal)+len(x_test_anomaly),3),dtype='U100')
for i in range(len(x_test_normal)):
    result[i,0] = str(os.path.basename(normal_path_test[i]))
    result[i,1] = 'OK'
    result[i,2] = str(Z1_arc[i])
for i in range(len(x_test_anomaly)):
    result[len(x_test_normal)+i,0] = str(os.path.basename(anomaly_path_test[i]))
    result[len(x_test_normal)+i,1] = 'NG'
    result[len(x_test_normal)+i,2] = str(Z2_arc[i])
np.savetxt('result.csv',result,delimiter=',',fmt='%s')

for i in range(x_test_anomaly.shape[0]):
    print(i,'/',x_test_anomaly.shape[0])
    test = read_and_preprocess_img(anomaly_path_test[i],size=(image_size,image_size))
    label = anomaly_label[i].reshape((1,2))
    img_heatmap = ScoreCam(model, label, test, -5)
    img_heatmap = superimpose(anomaly_path_test[i], img_heatmap, emphasize=True)
    img_heatmapname = "heatmap/"+os.path.basename(anomaly_path_test[i])
    cv2.imwrite(img_heatmapname, img_heatmap)

print("Completed.")
