from tensorflow.keras.models import load_model
import pandas as pd
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from scipy.spatial import distance
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from io import BytesIO
from PIL import Image as PILImage

def find_similar_images(new_image_features, features_dict, threshold=0.5):
    similarities = {}
    if new_image_features is not None:
        for cid, features in features_dict.items():
            if features is not None:
                sim = 1 - distance.cosine(new_image_features, features)
                if sim > threshold:
                    similarities[cid] = sim
    sorted_similar_images = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    return [img[0] for img in sorted_similar_images]

def extract_features(img_array):
    features = model2.predict(img_array)
    return features.flatten()
def load_and_preprocess_image(img_data):
    img = PILImage.open(BytesIO(img_data))
    img = img.resize((224, 224))  # Resize nếu cần
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Hàm để trích xuất đặc trưng mà không cần lưu file
def preprocess_and_extract_features(img_data):
    img_array = load_and_preprocess_image(img_data)
    features = model2.predict(img_array)
    return features.flatten()

def findImage(image_data):
    print("1")
    new_image_features = preprocess_and_extract_features(image_data)
    print("2")
    similar_images = find_similar_images(new_image_features, features_dict)
    print("3")
    print("Input img")
    print("Output img")
    images = []  # Khởi tạo images là một danh sách
    for img_path in similar_images[:5]:
        images.append(img_path)  # Thêm từng đường dẫn ảnh vào danh sách
    result = {
        "message": "Here are the similar images found",
        "type": "image_list",
        "images": images
    }

    # Chuyển đổi dictionary thành JSON
    json_result = json.dumps(result)
    return json_result

