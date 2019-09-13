import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import cv2
import keras
import glob
from PIL import Image
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
from keras.applications import MobileNetV2
from keras.layers import Input, GlobalAveragePooling2D, Activation
from keras.models import Model
from keras.optimizers import Adam
from sklearn import metrics
import time
import os

ok_train = 2832
ok_test = 100
ng_train = 8
ng_test = 1

class data:
    def __init__(self, data_size):
        self.data_size = data_size

    def resize(self, x, to_color=False):
        result = []

        for i in range(len(x)):
            if to_color:
                img = cv2.cvtColor(x[i], cv2.COLOR_GRAY2RGB)
                img = cv2.resize(img,dsize=(96,96))
            else:
                img = cv2.resize(x[i],dsize=(96,96))
            result.append(img)

        return np.array(result)

    def choose_data(self, x, y, normal_id, anomaly_id):
        x_normal, x_anomaly = [], []
        x_ref, y_ref = [], []
        j = 0
        for i in range(len(x)):
            if y[i] == normal_id:
                x_normal.append(x[i].reshape((x.shape[1:])))
            elif y[i] == anomaly_id:
                x_anomaly.append(x[i].reshape(x.shape[1:]))
                if j < ng_train:
                   x_ref.append(x[i].reshape((x.shape[1:])))
                   y_ref.append(y[i])
                   j += 1

        return np.array(x_normal), np.array(x_anomaly), np.array(x_ref), y_ref

    def get_data(self):
        oks = glob.glob('images/ok/*')
        ngs = glob.glob('images/ng/*')
        x_train = np.zeros((ok_train+ng_train,96,96,3),dtype=np.float32)
        y_train = np.zeros(ok_train+ng_train)
        x_test = np.zeros((ok_test+ng_test,96,96,3),dtype=np.float32)
        y_test = np.zeros(ok_test+ng_test)
        normal_path = []
        anormal_path = []
        for i in range(ok_train+ok_test):
           image = Image.open(oks[i])
           image = np.array(image)
           image = image[np.newaxis]
           image = self.resize(image)
           print(i,'/',ok_train+ok_test)
           if i < ok_train:
              x_train[i] = image
              y_train[i] = 0
           elif i < ok_train+ok_test:
              x_test[i-ok_train] = image
              y_test[i-ok_train] = 0
              normal_path.append(oks[i])
        for i in range(ng_train+ng_test):
           print(i,'/',ng_train+ng_test)
           image = Image.open(ngs[i])
           image = np.array(image)
           image = image[np.newaxis]
           image = self.resize(image)
           if i < ng_train:
              x_train[i+ok_train] = image
              y_train[i+ok_train] = 1
           elif i < ng_train+ng_test:
              x_test[i-ng_train+ok_test] = image
              y_test[i-ng_train+ok_test] = 1
              anormal_path.append(ngs[i])

        x_train = x_train / 255
        x_test = x_test / 255

        x_train_normal, _, x_ref, y_ref = self.choose_data(x_train, y_train, 0, 1)
        y_ref = to_categorical(y_ref)

        x_test_normal, x_test_anomaly, _, _ = self.choose_data(x_test, y_test, 0, 1)

        x_train_normal = self.resize(x_train_normal)
        x_ref = self.resize(x_ref)
        x_test_normal = self.resize(x_test_normal)
        x_test_anomaly = self.resize(x_test_anomaly)

        return x_train_normal, x_ref, y_ref, x_test_normal, x_test_anomaly, normal_path, anormal_path

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
    base_model=MobileNetV2(input_shape=x.shape[1:],alpha=0.5,
                           weights='imagenet',
                           include_top=False)

    c = base_model.output
    yinput = Input(shape=(classes,))
    hidden = GlobalAveragePooling2D()(c) 
    c = Arcfacelayer(classes, 30, 0.05)([hidden,yinput])
    prediction = Activation('softmax')(c)
    model = Model(inputs=[base_model.input, yinput], outputs=prediction)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0001, amsgrad=True),
                  metrics=['accuracy'])

    hist = model.fit([x, y], y, batch_size=128, epochs=10, verbose = False)

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

def Visualize(input_model, train, test, label_num):

    model = Model(input_model.get_layer(index=0).input, input_model.get_layer(index=-4).output)
#    model.summary()
    Test = np.expand_dims(test, axis=0)

    gradient_function = K.function([model.layers[0].input], [model.layers[-2].output])
    layer_output = gradient_function([Test])[0]

    G, R, ch = layer_output.shape[1:]
    res = np.zeros((G,R))

    for i in range(ch):
        img_res = layer_output[0,:,:,i]
        res = res + img_res

    res = res/ch

    res_flatte = np.ma.masked_equal(res,0)
    res_flatte = (res_flatte - res_flatte.min())*255/(res_flatte.max()-res_flatte.min())
    res_flatte = np.ma.filled(res_flatte,0)

    acm_img = cv2.applyColorMap(np.uint8(res_flatte), cv2.COLORMAP_JET)
    acm_img = cv2.cvtColor(acm_img, cv2.COLOR_BGR2RGB)
    acm_img = cv2.resize(acm_img,(96,96))

    jetcam = (np.float32(acm_img)*0.6 + test * 255 * 0.4)

    return np.uint8(jetcam)

DATA = data(ng_train)

x_train_normal, x_ref, y_ref, x_test_normal, x_test_anomaly, normal_path, anormal_path = DATA.get_data()
print(x_train_normal.shape,x_ref.shape,y_ref.shape,x_test_normal.shape,x_test_anomaly.shape)

normal_label = np.zeros((len(x_train_normal), 2))
normal_label[:,0] = 1

model_a = train_arcface(np.vstack((x_train_normal, x_ref)), np.vstack((normal_label, y_ref)), y_ref.shape[1])

t1 = time.time()
Z1_arc = -get_score_arc(model_a, x_train_normal, x_test_normal)
Z2_arc = -get_score_arc(model_a, x_train_normal, x_test_anomaly)

result = np.empty((ok_test+ng_test,2),dtype='U100')
for i in range(ok_test):
    result[i,0] = str(os.path.basename(normal_path[i]))
    result[i,1] = str(Z1_arc[i])
for i in range(ng_test):
    result[ok_test+i,0] = str(os.path.basename(anormal_path[i]))
    result[ok_test+i,1] = str(Z2_arc[i])
np.savetxt('result.csv',result,delimiter=',',fmt='%s')
t2 = time.time()
print('time',t2-t1)

for i in range(10):
    train = x_train_normal
    test = x_test_normal[i,:,:,:]
    img_GCAMplusplus = Visualize(model_a, train, test, 0)
    img_Gplusplusname = "heatmap/"+os.path.basename(normal_path[i])
    cv2.imwrite(img_Gplusplusname, img_GCAMplusplus)

for i in range(x_test_anomaly.shape[0]):
    train = x_ref
    test = x_test_anomaly[i,:,:,:]
    img_GCAMplusplus = Visualize(model_a, train, test, 1)
    img_Gplusplusname = "heatmap/"+os.path.basename(anormal_path[i])
    cv2.imwrite(img_Gplusplusname, img_GCAMplusplus)
print("Completed.")
