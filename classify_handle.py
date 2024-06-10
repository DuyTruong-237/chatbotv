import asyncio
import websockets
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import json



# Hàm phân loại câu hỏi
def classify_question_with_model(input_text, model, tokenizer, label_map):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=1).numpy()
    label_idx = np.argmax(probabilities)
    typequestion=label_map[label_idx]
    
    return typequestion
    # if typequestion=="product_information":
    #     print (typequestion)
    #     response = predict(input_text, model1, tokenizer1, device)
        
    # else:
    #     response = chat(input_text,chatbot,label2id,intents)
    # print(response)
    # result = {
    #     "message": response,
    #     "type": "genral"
        
    # }
    # # Chuyển đổi dictionary thành JSON
    # json_result = json.dumps(result)
    # return json_result
    # return response  # Trả về tag dựa vào chỉ số nhãn
