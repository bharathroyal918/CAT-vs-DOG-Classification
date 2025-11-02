import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load trained model
model = load_model('model/cat_dog_classifier.h5')

# Path to test image
img_path = 'C:/Users/bhara/Documents/Cat vs Dog Classification/test1/test1/164.jpg'  # or 'my_dog.jpg'

# Preprocess image
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Show image
plt.imshow(image.load_img(img_path))
plt.axis('off')
plt.show()

# Predict
prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("🐶 It's a Dog")
else:
    print("🐱 It's a Cat")
