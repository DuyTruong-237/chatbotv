import sys
from transformers import pipeline, BertForSequenceClassification, BertTokenizerFast
import random
import json
import pandas as pd
import asyncio
import websockets
from functools import partial

def load_json_file(filename):
    with open(filename) as f:
        file = json.load(f)
    return file

def create_df():
    df = pd.DataFrame({
        'Pattern' : [],
        'Tag' : []
    })
    return df

def extract_json_info(json_file, df):
    for intent in json_file['intents']:
        for pattern in intent['patterns']:
            sentence_tag = [pattern, intent['tag']]
            df.loc[len(df.index)] = sentence_tag
    return df

def load_chatbot(model_path):
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    chatbot = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return chatbot

from collections import Counter
            
def find_most_similar(user_input, responses):
    # Sử dụng Counter để đếm số lần xuất hiện của từng từ trong câu người dùng
    user_words = Counter(user_input.lower().split())
    
    max_matches = 0
    best_response = None
    
    # Kiểm tra từng câu trong danh sách câu trả lời
    for response in responses:
        response_words = Counter(response.lower().split())
        # Tính số từ trùng khớp sử dụng phép giao của hai Counter
        matches = sum((user_words & response_words).values())
        
        # Cập nhật câu trả lời tốt nhất nếu tìm thấy nhiều từ trùng khớp hơn
        if matches > max_matches:
            max_matches = matches
            best_response = response
            
    return best_response
def chat(question,chatbot,label2id,intents):
    label = label2id[chatbot(question)[0]['label']]
    print ("hio")
    response = find_most_similar(question,intents['intents'][label]['responses'])
    return response
filename = './intend_number.json'
intents = load_json_file(filename)
df = create_df()
df2 = extract_json_info(intents, df)
labels = df2['Tag'].unique().tolist()
labels = [s.strip() for s in labels]
label2id = {label: id for id, label in enumerate(labels)}
model_path = "./Support_final"
chatbot = load_chatbot(model_path)

# async def chat_handler(websocket, path,chatbot,label2id,intents):
#     async for message in websocket:
#         print(f"Received message: {message}")
#         # Your chat logic here
#         response = chat(message,chatbot,label2id,intents)
#         await websocket.send(response)

# async def main():
#     filename = './intents.json'
#     intents = load_json_file(filename)
#     df = create_df()
#     df2 = extract_json_info(intents, df)
#     labels = df2['Tag'].unique().tolist()
#     labels = [s.strip() for s in labels]
#     label2id = {label: id for id, label in enumerate(labels)}
#     model_path = "./"
#     chatbot = load_chatbot(model_path)
#     async with websockets.serve(partial(chat_handler,chatbot=chatbot, label2id=label2id, intents=intents), "localhost", 6789):
#         await asyncio.Future()  # Run forever

# if __name__ == '__main__':
#      asyncio.run(main())












from transformers import BertTokenizer, BertForTokenClassification, BertConfig
import torch
import asyncio
import websockets
from functools import partial
import mysql.connector
import json
from decimal import Decimal
from json import JSONEncoder

class DecimalEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)  # or `float(obj)` if precision is less critical
        # Let the base class default method raise the TypeError
        return JSONEncoder.default(self, obj)

# Replace the below connection parameters with your details
config = {
    'user': 'root',
    'password': '123456',
    'host': 'localhost',
    'database': 'shop',
    'raise_on_warnings': True
}

try:
    # Establishing the connection
    cnx = mysql.connector.connect(**config)

    # Do something with the connection
    print("Connected to the database successfully!")

except mysql.connector.Error as err:
    print("An error occurred: {}".format(err))

# finally:
#     # Closing the connection
#     if cnx.is_connected():
#         cnx.close()
#         print("Connection closed.")

# Global variables
model1_path = 'best_model_state2.bin'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer1 = BertTokenizer.from_pretrained('bert-base-uncased')
config1 = BertConfig.from_pretrained('bert-base-uncased', num_labels=13)
model1 = BertForTokenClassification.from_pretrained("bert-base-uncased", config=config1)
model1.load_state_dict(torch.load(model1_path), strict=False)
model1.to(device)
model1.eval()
def predict(input_text, model1, tokenizer, device):
    # Tokenize the input data
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    print(input_text)
    # Perform prediction
    with torch.no_grad():
        outputs = model1(**inputs)
        predictions = outputs[0]

    # Convert logits to probabilities and then to labels
    probabilities = torch.nn.functional.softmax(predictions, dim=-1)
    predicted_label = probabilities.argmax(dim=-1)
    return extract_entities(input_text, predicted_label.tolist())
def extract_entities(input_text, label_ids):
    # Tokenize the input text and align labels with tokens
    tokenized_input = tokenizer1.tokenize(input_text)
    print("Tokens:", tokenized_input)
    print("Labels:", label_ids)

    # Dictionary to hold the extracted entities
    entities = {
        "category": [],
        "color": [],
        "size": [],
        "gender": [],
        "price": []
    }

    # Helper function to get entity name from label id
    def get_entity_type(label_id):
        if label_id == 1:
            return "category"
        elif label_id == 3:
            return "color"
        elif label_id == 5:
            return "size"
        elif label_id == 7:
            return "gender"
        elif label_id == 9:
            return "price"
        return None

    # Extract entities based on labels
    for token, label_id in zip(tokenized_input, label_ids[0]):  # assuming label_ids is a 2D list [[...]]
        entity_type = get_entity_type(label_id)
        if entity_type:
            entities[entity_type].append(token)

    # Convert lists to strings
    for key in entities:
        entities[key] = " ".join(entities[key])

    # Remove empty keys
    entities = {key: value for key, value in entities.items() if value}
    print(entities,"hoiiiiii")

    conditions = " AND ".join([f"{key} = '{value}'" for key, value in entities.items()])
    res = " , ".join([f"{key} is '{value}'" for key, value in entities.items()])
    res1= "This is result "+ res

# SQL query
    query = f"SELECT * FROM Shoe WHERE {conditions}"
    print(query)
    cursor = cnx.cursor(dictionary=True)  # Use dictionary cursor to ease JSON conversion

    # Execute the query
    cursor.execute(query)

    # Fetch all the rows
    results = cursor.fetchall()

    # Convert the result into JSON format
    #json_result = json.dumps(results)
    json_result = json.dumps(results, cls=DecimalEncoder)
    return res1+"\n" + json_result

# async def chat_handler(websocket, path, model1, tokeni678zer, device):
#     async for message in websocket:
#         print(message)
#         response = await predict(message, model1, tokenizer, device)
#         await websocket.send(str(response))

# async def main():
#     async with websockets.serve(partial(chat_handler, model1=model1, tokenizer=tokenizer, device=device), "localhost", 6789):
#         await asyncio.Future()  # This will run forever

# if __name__ == '__main__':
#     asyncio.run(main())

from tensorflow.keras.models import load_model
import pandas as pd
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from scipy.spatial import distance

try:
    model2 = load_model('model2.h5')
    features_dict = np.load('features_dict2.npy', allow_pickle=True).item()
except Exception as e:
    print("Error:", e)

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

# def preprocess_and_extract_features(image_path):
#     img = load_and_preprocess_image(image_path)
#     if img is not None:
#         features = extract_features(img)
#         return features
#     return None

# data = pd.read_csv('./meta-data1.csv')
# data['CID'] = data['CID'].apply(lambda x: x + '.jpg')

# def load_and_preprocess_image(img_path):
#     img = image.load_img(img_path, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array)
#     return img_array

# def findImage(image_data):
#     print("1")
#     new_image_features = preprocess_and_extract_features(image_data)
#     print("2")

#     similar_images = find_similar_images(new_image_features, features_dict)
#     print("3")

#     from IPython.display import Image, display
#     print("Input img")
#     print("Output img")
#     images = ""
#     for img_path in similar_images[:5]:
#         images += img_path
#     print (images)
#     return images
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from io import BytesIO
from PIL import Image as PILImage

# Giả sử hàm này là để tải và tiền xử lý ảnh từ dữ liệu
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
    print(json_result)
    return json_result



import asyncio
import websockets
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import json

# Hàm tải mô hình và tokenizer
def load_model_and_tokenizer(model_path):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer

# Hàm phân loại câu hỏi
def classify_question_with_model(input_text, model, tokenizer, label_map):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=1).numpy()
    label_idx = np.argmax(probabilities)
    typequestion=label_map[label_idx]
    print(typequestion, input_text)
    if typequestion=="product_information":
        print (typequestion)
        response = predict(input_text, model1, tokenizer1, device)
        
    else:
        response = chat(input_text,chatbot,label2id,intents)
    print(response)
    result = {
        "message": response,
        "type": "genral"
        
    }

    # Chuyển đổi dictionary thành JSON
    json_result = json.dumps(result)
    return json_result
    return response  # Trả về tag dựa vào chỉ số nhãn

# Tạo bản đồ nhãn
label_map = {0: "general_question", 1: "product_information", 2: "ambiguous_questions"}

# Load mô hình và tokenizer
model_path = './Classify'  # Thay đổi đường dẫn này thành thư mục chứa mô hình và tokenizer
model, tokenizer = load_model_and_tokenizer(model_path)
import base64
import asyncio

import os
from datetime import datetime
# async def classify_question(websocket, path):
#     async for message in websocket:
      
#         # Kiểm tra xem tin nhắn có phải là ảnh không
#         if message.startswith("data:image/"):
#             # Xử lý ảnh tại đây, ví dụ lưu trữ ảnh hoặc gửi cho một hàm khác xử lý
#             try:
#                 header, encoded = message.split(",", 1)
#                 image_data = base64.b64decode(encoded)
#                 # Tạo một tên file duy nhất cho ảnh
#                 img_format = header.split(';')[0].split('/')[1]
#                 filename = f"images/{datetime.now().strftime('%Y%m%d%H%M%S')}.{img_format}"
#                 await websocket.send("wait")
#                 # Lưu ảnh vào một thư mục
#                 os.makedirs('images', exist_ok=True)
#                 with open(filename, 'wb') as f:
#                     f.write(image_data)
#                 response=findImage(filename)
#                 print ("response: " + response)
#                 await websocket.send(response)
#             except Exception as e:
#                 print("Error handling image:", e)
#                 await websocket.send("Error processing image.")

#         else:
#             # Xử lý tin nhắn văn bản như thông thường
#             print("Message",message)
#             response = classify_question_with_model(message, model, tokenizer, label_map)
#             await websocket.send(str(response))
import base64
import os

from datetime import datetime
import aiofiles
import os

async def save_image_async(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    async with aiofiles.open(path, 'wb') as f:
        await f.write(data)

async def classify_question(websocket, path):
    async for message in websocket:
        if message.startswith("data:image/"):
            try:
                header, encoded = message.split(",", 1)
                image_data = base64.b64decode(encoded)
                img_format = header.split(';')[0].split('/')[1]
                filename = f"images/{datetime.now().strftime('%Y%m%d%H%M%S')}.{img_format}"
                await websocket.send("Processing image, please wait...")
                #await save_image_async(filename, image_data)
                response = findImage(image_data)
                await websocket.send(response)
            except Exception as e:
                print("Error handling image:", e)
                await websocket.send(f"Error processing image: {str(e)}")
        else:
            print("Message", message)
            response = classify_question_with_model(message, model, tokenizer, label_map)
            await websocket.send(str(response))

# async def classify_question(websocket, path):
#     async for message in websocket:
       
#         response = classify_question_with_model(message, model, tokenizer, label_map)
#         await websocket.send(str(response))

# Hàm khởi chạy WebSocket server
async def main():
    async with websockets.serve(classify_question, "localhost", 6789):
        await asyncio.Future()  # This will run forever

# Chạy WebSocket server
asyncio.run(main())




