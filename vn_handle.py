import pandas as pd
import json
from underthesea import word_tokenize, text_normalize
import numpy as np
import random
from transformers import PhobertTokenizer, RobertaForSequenceClassification
import torch


def preprocess_text(text):
    normalized_text = text_normalize(text)
    tokens = word_tokenize(normalized_text)
    return " ".join(tokens)

def load_dataset(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data['intents']

def classify_question_vn(input_text, dataset, model,tokenizer, id2label):
    input_text = preprocess_text(input_text.lower())
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    tag = id2label[prediction]  # Use id2label to get the tag
    return respond_to_question(tag, dataset)

# def respond_to_question(tag, dataset):
#     for intent in dataset:
#         if intent['tag'] == tag:
#             return random.choice(intent['responses'])
#     return "Xin lỗi, tôi không hiểu câu hỏi của bạn."


def respond_to_question(tag, dataset):
    respond="Xin lỗi, tôi không hiểu câu hỏi của bạn."
    

    for intent in dataset:
        if intent['tag'] == tag:
            respond= random.choice(intent['responses'])
            break
    result={
            "message":respond,
            "type":"add success",
            "data":""
        }
   
    return json.dumps(result)



