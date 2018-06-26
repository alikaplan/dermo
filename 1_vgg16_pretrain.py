from keras.applications import VGG16, InceptionV3
from keras.preprocessing import image# image generetor resimlerin ön işleme kısmı için gerkli kutuphane(resimleri alıp
#  matrislere cevirip yeniden boyutlandırıp matrise aktarıyor)
from keras.layers import GlobalAveragePooling2D, Dense, Dropout#Dense  100 tane noron koy mesela , dropout 100 tane
# noranda nyuzde 30 unu alma dıyoruz overfiting engellensin diye

from keras.models import Model#medeli olusturan ana class
from keras import optimizers # optimizayon algoritmaları vaid ve baes için güncelleme için gerekli optimizsayonu
#  yapan algortmalar,gradiyent discent
from keras.callbacks import TensorBoard, ModelCheckpoint#tensorboard loglama validation acr. felan onları tutuoyr neydı ne oldu dıye
#modelcheckpoint kaldıgı yerden devame tmesini saglayan classlar
import argparse #cmd üzerinden klasor isimlerini felan alamaya yarayan kısmı bunun sayesınden gerekli yolları ve arg veriyor
from time import time # ne akdar suuruyor onun ıcı ?
import os # işletim sistemi ile alakalı kutuphane

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"#loglamaayıo mesela azaltıyor loglama derecesı artıtırsan daha cok asmaamın bılgısını verıyor
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'# otomatık gpu da calısmasını engellemek ııcın


img_size=224# modele gonderecegımız resımlerın en ve boyunu belırlıyoruz 224*244 verecegız vgg16 buna gore yapıdıgı ıcın

ap = argparse.ArgumentParser()
ap.add_argument("-train","--train_dir",type=str, required=True,help="(gerekli) egıtım klasorunu gırınız")
ap.add_argument("-val","--val_dir",type=str, required=True,help="(required) the validation data directory")
ap.add_argument("-num_class","--class",type=int, default=2,help="(required) number of classes to be trained")#egıtılmesını ıstedıgınız tur sayısıbızde 2 kanserlı degıl ?

args = vars(ap.parse_args())#usttekı tanımlamarı bır dızı at ve tanımlama olarak args ıcıne at

##### Step-2: veri hazırlama

batch_size=32# kaçar kaçar resımlerı modele verecegız onu secıyoruz

train_datagen = image.ImageDataGenerator(#train dataları ıcın  olustrulmus bır class resımlerı sag sol yap kucult gıbı ıslemler ıcın reısm adetını arttırmak ıcın
#        width_shift_range=0.1,
#        height_shift_range=0.1,
        rescale=1./255,# 01 arasına al rgb 2555 e kadar olan degerlerı 1 ve 0 arasına alıyoruz rahat ıslem ıcın
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

valid_datagen = image.ImageDataGenerator(rescale=1. / 255)


train_generator = train_datagen.flow_from_directory(
        args["train_dir"],
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical') # klasorları klasorledıgımız ıcın kadergorık oalrak ayırdık bundan haberı olsun dıye bu satır

validation_generator = valid_datagen.flow_from_directory(
        args["val_dir"],
        target_size=(img_size,img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)# resımlerı karıstır oyle al


##### Step-3:Model olusturma


print('model ve onceden egıtılmıs verı yuklenıyor...')

base_model = VGG16(include_top=False, weights='imagenet')# include top vgg1 ya dokunma ben kendıme gore duzenleyecegım
#base_model = InceptionV3(include_top=False, weights='imagenet')

i=0
for layer in base_model.layers:
    layer.trainable = False
    i = i+1
    print(i,layer.name)# bu dongu tum katanları degısımlerını false yapıyor egıtımı durduruyor.

##### Step-4: modelın son kısmını sılıp kendı problemımıze gore duzenleme


x = base_model.output # x atmaa yap modelı
x = Dense(128)(x) # x katmanına 128 noron ekmleme
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x) #128 noronun yuzde 20 sını egıtımde devre dısı bırak
predictions = Dense(args["class"], activation='softmax')(x)#


##### Step-5:


tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

filepath = 'dermatology_pretrained_model.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,save_best_only=True,save_weights_only=False, mode='min',period=1)
#modelchckpoınt clası her epohtakı egıtım de calısmalrı kaydedıyor loss degerı accr felan
callbacks_list = [checkpoint,tensorboard]



model = Model(inputs=base_model.input, outputs=predictions)# modelın gırısı vgg16 cıkısı predic. diyoruz

#model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.001, momentum=0.9),metrics=["accuracy"])
model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(),metrics=["accuracy"])# kayıp deger ve metrık
#model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0),metrics=["accuracy"])

num_training_img=17341
num_validation_img=2166
stepsPerEpoch = num_training_img/batch_size # 1 epoch  tamamlaayabılmesı ıcın atması gereken adım
validationSteps= num_validation_img/batch_size

model.fit_generator(# calıstırma fonksıyonu
        train_generator,
        steps_per_epoch=stepsPerEpoch,
        epochs=20,
        callbacks = callbacks_list,
        validation_data = validation_generator,
        validation_steps=validationSteps,
        )


