import sys
from transformers import pipeline, BertForSequenceClassification, BertTokenizerFast,BertTokenizer, BertForTokenClassification, BertConfig
import random
import json
from functools import partial
import mysql.connector
from json import JSONEncoder
from decimal import Decimal
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from scipy.spatial import distance
import torch
import base64
import os
from json import JSONEncoder

from decimal import Decimal
from datetime import datetime
import aiofiles
import os
import pandas as pd
import asyncio
import websockets
from functools import partial
from load_model import (load_chatbot_model,load_model_and_tokenizer_entities, load_token_classification_model, 
                          load_image_classification_model, load_question_classification_model,load_model_vn,extract_json_info_vn,create_df_vn,load_json_file_vn,load_dataset_vn)
from support_handle import (chat, load_json_file, create_df,extract_json_info)
from enttities_handle import (checkoutHandle,predict,add_to_cart_handle,find_product,find_products)
from cnn_handle import (find_similar_images, extract_features,load_and_preprocess_image, preprocess_and_extract_features,findImage)
from classify_handle import (classify_question_with_model)
from vn_handle import (classify_question_vn)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Đường dẫn tới các mô hình
chatbot_model_path = "./model/support"
token_classification_model_path = './model/best_model_state2.bin'
image_classification_model_path = './model/model1.h5'
question_classification_model_path = './model/Classify2'
model_path_vn = "./model/VN_Support1"
filename_vn = './model/vn_intend.json'
# Load các mô hình
chatbot = load_chatbot_model(chatbot_model_path)
model1, tokenizer1,label_dict = load_model_and_tokenizer_entities(device)

model3, tokenizer3 = load_question_classification_model(question_classification_model_path)
# def load_dataset(filename):
#     with open(filename, 'r', encoding='utf-8') as file:
#         data = json.load(file)
#     return data['intents']
# # modelvn = BertForSequenceClassification.from_pretrained("VN_Support1")
# # tokenizervn = BertTokenizer.from_pretrained("VN_Support1")
# datasetvn = load_dataset('./model/vn_intend.json')



filename = './model/intend_number_final.json'
intents = load_json_file(filename)
df = create_df()
df2 = extract_json_info(intents, df)
labels = df2['Tag'].unique().tolist()
labels = [s.strip() for s in labels]
label2id = {label: id for id, label in enumerate(labels)}


intents_vn = load_json_file_vn(filename_vn)
df_vn = create_df_vn()
df2_vn = extract_json_info_vn(intents_vn, df_vn)
labels_vn = df2_vn['Tag'].unique().tolist()
labels_vn = [s.strip() for s in labels_vn]
num_labels_vn = len(labels_vn)
label2id_vn = {label: id for id, label in enumerate(labels_vn)}
id2label_vn = {id:label for id, label in enumerate(labels_vn)}
dataset_vn = load_dataset_vn('./model/vn_intend.json')
model_vn, tokenizer_vn=load_model_vn(model_path_vn, num_labels_vn, id2label_vn, label2id_vn)
class DecimalEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return str(obj)  # or `float(obj)` if precision is less critical
        # Let the base class default method raise the TypeError
        return JSONEncoder.default(self, obj)

# Replace the below connection parameters with your details
config = {
    'user': 'root',
    'password': 'root',
    'host': 'localhost',
    'database': 'shop4',
    'port': 3308,
    'raise_on_warnings': True
}
try:
    # Establishing the connection
    cnx = mysql.connector.connect(**config)

    # Do something with the connection
    print("Connected to the database successfully!")

except mysql.connector.Error as err:
    print("An error occurred: {}".format(err))

label_map = {0: "support_request", 1: "handle_request", 2: "ambiguous_questions"}

sessions_data = {}
def handle_Q(message, model3, tokenizer3, label_map):
    print("Processing message...")
    typequestion = classify_question_with_model(message, model3, tokenizer3, label_map)
    if typequestion == "handle_request":
        response = predict(message,model1, tokenizer1, device) 
        try:
            response = json.loads(response) if isinstance(response, str) else response
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            response = {}
    else: response=chat(message, chatbot, label2id, intents)
   
    print("type of the message '" + message+"' is "+ typequestion)



    return response,typequestion
def custom_json_serializer(obj):
    if isinstance(obj, Decimal):
        return float(obj) 
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
def goHandle(data):

        if(data['intent'][0]==2):
            if(data['intent'][-1]==0):
                return find_product(data,cnx)

            return add_to_cart_handle(data,cnx)
        else:
            if(data['intent'][-1]==2):
                return add_to_cart_handle(data,cnx)
            elif(data['intent'][-1]==5):
                return find_products(data,cnx)
            elif(data['intent'][-1]==8):
                return checkoutHandle(data)
            elif(data['intent'][-1]==0):
                return find_product(data,cnx)
            else:
                return find_products(data,cnx)
        # if (len(data['intent'])>1):
        #     if(data['intent'][-1]==7):
        #         return add_to_cart_handle(data,cnx)
        #     else:
        #         if(data['intent'][-1]==0):
        #             if(data['intent'][-2]==7):
        #                 return add_to_cart_handle(data,cnx)
        #         return find_products(data,cnx)
            
        # else:
        #     if(data['intent'][-1]==7):
        #         return add_to_cart_handle(data,cnx)
        #     else:
        #         return  find_products(data,cnx)
async def classify_question(websocket, path):
    session_id = str(websocket.id)
    if session_id not in sessions_data:
        sessions_data[session_id] = {"message": "Received message", "type": "general", "intent": [], "entities": {}}

    async for message in websocket:
        if message.startswith("#V"):
            message = message.replace("#V", "")
            response= classify_question_vn(message, dataset_vn, model_vn,tokenizer_vn, id2label_vn)
            print(response)
            await websocket.send(response)
        else:
            message = message.replace("#E", "")
            response_data, typeQuestion = handle_Q(message, model3, tokenizer3, label_map)
            if(typeQuestion=="handle_request"):
                current_session = sessions_data[session_id]

                # Update intent predictions
                current_session['intent'].extend(response_data.get("intent_prediction", []))
                print("intent", current_session['intent'] 
                )
                # Update entities
                entities = response_data.get('entities', {}) 
                current_session_entities = current_session['entities']
                current_session_entities['category'] = entities.get('category', current_session_entities.get('category'))
                current_session_entities['color'] = entities.get('color', current_session_entities.get('color'))
                current_session_entities['name'] = entities.get('name', current_session_entities.get('name'))
                current_session_entities['size'] = entities.get('size', current_session_entities.get('size'))
                current_session_entities['price'] = entities.get('price', current_session_entities.get('price'))
                print(sessions_data[session_id])
                response= goHandle(sessions_data[session_id])
                print("ttttt")
                print(response["type"])
                if(response["type"]!="errol" and response["type"]!="product_info"):
                    current_session['intent'] = []
                print("intent", current_session['intent'] 
                )
                
                response= json.dumps(response, default=custom_json_serializer, ensure_ascii=False, indent=4)


                # Send updated session data back to client
              
            else: 
                response= response_data
            
            await websocket.send(response)
            

async def main():
    async with websockets.serve(classify_question, "localhost", 6789):
        await asyncio.Future()  # This will run forever


   
if __name__ == "__main__":
    asyncio.run(main())