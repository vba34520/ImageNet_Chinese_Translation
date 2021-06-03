import numpy as np
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.python.keras.preprocessing import image

model = mobilenet_v2.MobileNetV2(weights='imagenet')
img = image.load_img('dog.jpg', target_size=(224, 224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = mobilenet_v2.preprocess_input(img)
print(img.shape)
pred_class = model.predict(img)
n = 10
top_n = mobilenet_v2.decode_predictions(pred_class, top=n)
for c in top_n[0]:
    print(c)
