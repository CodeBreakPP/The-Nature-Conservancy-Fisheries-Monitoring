import os
os.chdir('Desktop/input')
from utils import *
from vgg16bn import Vgg16BN

path=""
batch_size=64
(val_classes, trn_classes, val_labels, trn_labels,val_filenames, filenames, test_filenames) = get_classes(path)

trn = get_data(path+'train', (540,960))
val = get_data(path+'valid', (540,960))
test = get_data(path+'test', (540,960))
vgg640 = Vgg16BN((540,960)).model
vgg640.pop()
vgg640.input_shape, vgg640.output_shape
vgg640.compile(Adam(), 'categorical_crossentropy', metrics=['accuracy'])
##
conv_val_feat = vgg640.predict(val, batch_size=32, verbose=1)
conv_trn_feat = vgg640.predict(trn, batch_size=32, verbose=1)
conv_test_feat = vgg640.predict(test, batch_size=32, verbose=1)
##
save_array(path+'results/conv_val_960.dat', conv_val_feat)
save_array(path+'results/conv_trn_960.dat', conv_trn_feat)
save_array(path+'results/conv_test_960.dat', conv_test_feat)
##
conv_val_feat = load_array(path+'results/conv_val_960.dat')
conv_trn_feat = load_array(path+'results/conv_trn_960.dat')
conv_test_feat = load_array(path+'results/conv_test_960.dat')

conv_layers,_ = split_at(vgg640, Convolution2D)
nf=128; p=0.


def get_lrg_layers():
    return [
        BatchNormalization(axis=1, input_shape=conv_layers[-1].output_shape[1:]),
        Convolution2D(nf,3,3, activation='relu', border_mode='same'),
        BatchNormalization(axis=1),
        Convolution2D(nf,3,3, activation='relu', border_mode='same'),
        BatchNormalization(axis=1),
        Convolution2D(nf,3,3, activation='relu', border_mode='same'),
        BatchNormalization(axis=1),
        Convolution2D(8,3,3, border_mode='same'),
        GlobalAveragePooling2D(),
        Activation('softmax')
    ]

lrg_model = Sequential(get_lrg_layers())

lrg_model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

lrg_model.fit(conv_trn_feat, trn_labels, batch_size=batch_size, nb_epoch=10,validation_data=(conv_val_feat, val_labels))

lrg_model.load_weights(path+'models/lrg_0mp960.h5')
lrg_model.evaluate(conv_val_feat, val_labels)
preds = lrg_model.predict(conv_test_feat, batch_size=64)
preds=preds[1]
classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

subm = do_clip(preds,0.82)
submission = pd.DataFrame(subm, columns=classes)
submission.insert(0, 'image', raw_test_filenames)
submission.head()
submission.to_csv("submission2.26.21.18.csv", index=False)
