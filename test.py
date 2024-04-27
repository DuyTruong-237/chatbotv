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

def chat(question,chatbot,label2id,intents):
    label = label2id[chatbot(question)[0]['label']]
    response = random.choice(intents['intents'][label]['responses'])
    return response
filename = './intents1.json'
intents = load_json_file(filename)
df = create_df()
df2 = extract_json_info(intents, df)
labels = df2['Tag'].unique().tolist()
labels = [s.strip() for s in labels]
label2id = {label: id for id, label in enumerate(labels)}
model_path = "./Support"
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

# Global variables
model1_path = 'best_model_state.bin'
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

    return entities

# async def chat_handler(websocket, path, model1, tokenizer, device):
#     async for message in websocket:
#         print(message)
#         response = await predict(message, model1, tokenizer, device)
#         await websocket.send(str(response))

# async def main():
#     async with websockets.serve(partial(chat_handler, model1=model1, tokenizer=tokenizer, device=device), "localhost", 6789):
#         await asyncio.Future()  # This will run forever

# if __name__ == '__main__':
#     asyncio.run(main())

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
        print("s")
    else:
        response = chat(input_text,chatbot,label2id,intents)
    print(response)
    return response  # Trả về tag dựa vào chỉ số nhãn

# Tạo bản đồ nhãn
label_map = {0: "general_question", 1: "product_information", 2: "ambiguous_questions"}

# Load mô hình và tokenizer
model_path = './Classify'  # Thay đổi đường dẫn này thành thư mục chứa mô hình và tokenizer
model, tokenizer = load_model_and_tokenizer(model_path)

async def classify_question(websocket, path):
    async for message in websocket:
       
        response = classify_question_with_model(message, model, tokenizer, label_map)
        await websocket.send(str(response))

# Hàm khởi chạy WebSocket server
async def main():
    async with websockets.serve(classify_question, "localhost", 6789):
        await asyncio.Future()  # This will run forever

# Chạy WebSocket server
asyncio.run(main())
