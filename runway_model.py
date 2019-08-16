from keras import applications
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
import runway

@runway.setup
def setup():
  return applications.resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg')

@runway.command('extract_features', inputs={'image': runway.image}, outputs={'features': runway.vector(2048)})
def extract_features(model, inputs):
  x = np.array(inputs['image'].resize((224, 224)))
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  features = model.predict(x)[0]
  return features.flatten()

if __name__ == "__main__":
  runway.run()
