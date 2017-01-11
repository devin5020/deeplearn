# Deeplearn
based on python+ theano + keras

+ using theano backends

from scipy import misc
import copy
import numpy as np
from vggface import VGGFace

model = VGGFace()

im = misc.imread('../image/ak.jpg')
im = misc.imresize(im, (224, 224)).astype(np.float32)
aux = copy.copy(im)
im[:, :, 0] = aux[:, :, 2]
im[:, :, 2] = aux[:, :, 0]
# Remove image mean
im[:, :, 0] -= 93.5940
im[:, :, 1] -= 104.7624
im[:, :, 2] -= 129.1863
im = np.transpose(im, (2, 0, 1))  # need rgb transpose
im = np.expand_dims(im, axis=0)

res = model.predict(im)
print np.argmax(res[0])

