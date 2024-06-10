import sys
from transformers import pipeline, BertForSequenceClassification, BertTokenizerFast,BertTokenizer, BertForTokenClassification, BertConfig
import random
import json
from functools import partial
import mysql.connector
from json import JSONEncoder
from decimal import Decimal

import pandas as pd
import asyncio
import websockets
from functools import partial
def chat(question,chatbot,label2id,intents):
    # try:
    #     label = label2id[chatbot(question)[0]['label']]
    #     print ("hio")
    #     response = random.choice(intents['intents'][label]['responses'])
    #     result={
    #             "message":response,
    #             "type":"supmessage",

    #         }
    # catch:
    #     result={
    #             "message":"I don't understand",
    #             "type":"supmessage",

    #         }
    try:
        label = label2id[chatbot(question)[0]['label']]
        print("hio")
        response = random.choice(intents['intents'][label]['responses'])
        result = {
            "message": response,
            "type": "supmessage",
        }
    except Exception as e:
        print(f"An error occurred: {e}")  # In thông báo lỗi để dễ dàng gỡ lỗi
        result = {
            "message": "I don't understand",
            "type": "supmessage",
        }
    return json.dumps(result, indent=4)
   

    
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