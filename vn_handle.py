import pandas as pd
import json
from underthesea import word_tokenize, text_normalize
import numpy as np
import random
from transformers import BertTokenizer, BertForSequenceClassification
import torch



def preprocess_text(text):
    normalized_text = text_normalize(text)
    tokens = word_tokenize(normalized_text)
    return " ".join(tokens)

def load_dataset(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data['intents']

def classify_questionvn(input_text, dataset ):
    input_text = preprocess_text(input_text.lower())
    input_tokens = set(input_text.split())
    best_tag = None
    highest_similarity = -1
    for item in dataset:
        for pattern in item['patterns']:
            pattern_text = preprocess_text(pattern.lower())
            pattern_tokens = set(pattern_text.split())
            similarity = len(input_tokens.intersection(pattern_tokens)) / len(input_tokens.union(pattern_tokens))
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_tag = item['tag']
    return respond_to_question(best_tag, dataset)


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



