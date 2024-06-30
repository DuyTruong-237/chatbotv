from transformers import BertForSequenceClassification, BertTokenizer, BertTokenizerFast, pipeline, BertConfig,BertModel, BertPreTrainedModel,PhobertTokenizer, RobertaForSequenceClassification
from tensorflow.keras.models import load_model
import torch
import torch.nn as nn
import pandas as pd
import json

from transformers import BertForTokenClassification

intent_dict = {'checkout': 0, 'order_inquiry': 1, 'find': 2, 'cancel order': 3, 'update cart': 4, 'add to cart': 5, 'remove item': 6, 'undefined': 7, 'find one': 8, "default":9}
# Dictionary for NER labels
label_dict = {
    'O': 0,
    'B-color': 1, 'I-color': 2,
    'B-size': 3, 'I-size': 4,
    'B-name': 5, 'I-name': 6,
    'B-gender': 7, 'I-gender': 8,
    'B-price': 9, 'I-price': 10,
    'B-order': 11, 'I-order': 12,
    'B-pID': 13, 'I-pID': 14,
    'B-category': 15, 'I-category': 16,
    'B-quality': 17, 'I-quality': 18,
    'B-material': 19, 'I-material': 20,
    'B-product_feature': 21, 'I-product_feature': 22,
    'B-season': 23, 'I-season': 24,
    'B-quantity': 25, 'I-quantity': 26
}
class BertForJointIntentAndNER(BertPreTrainedModel):
    def __init__(self, config, num_intents, num_labels):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.intent_classifier = nn.Linear(config.hidden_size, num_intents)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, labels=None, intent_label=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0])
        pooled_output = self.dropout(outputs[1])

        logits = self.classifier(sequence_output)
        intent_logits = self.intent_classifier(pooled_output)

        loss = None
        if labels is not None and intent_label is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels))
            loss = loss_fct(active_logits, active_labels) + loss_fct(intent_logits.view(-1, self.num_intents), intent_label.view(-1))

        return {'loss': loss, 'logits': logits, 'intent_logits': intent_logits}

# def load_model_and_tokenizer_entities( model_path):
#     model_name='bert-base-uncased'
#     num_labels=45
#     num_intents=8
#     label_dict = {
#     'O': 0,
#     'B-color': 1, 'I-color': 2,
#     'B-size': 3, 'I-size': 4,
#     'B-name': 5, 'I-name': 6,
#     'B-gender': 7, 'I-gender': 8,
#     'B-price': 9, 'I-price': 10,
#     'B-order': 11, 'I-order': 12,
#     'B-pID': 13, 'I-pID': 14,
#     'B-category': 15, 'I-category': 16,
#     'B-quality': 17, 'I-quality': 18,
#     'B-material': 19, 'I-material': 20,
#     'B-product_feature': 21, 'I-product_feature': 22,
#     'B-season': 23, 'I-season': 24,
#     'B-quantity': 25, 'I-quantity': 26
#     }
#     tokenizer = BertTokenizer.from_pretrained(model_name)
    
#     class BertForNERAndIntent(torch.nn.Module):
#         def __init__(self, model_name, num_labels, num_intents):
#             super(BertForNERAndIntent, self).__init__()
#             self.bert = BertModel.from_pretrained(model_name)
#             self.dropout = torch.nn.Dropout(0.1)
#             self.ner_classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)
#             self.intent_classifier = torch.nn.Linear(self.bert.config.hidden_size, num_intents)

#         def forward(self, input_ids, attention_mask, labels=None, intent_labels=None):
#             outputs = self.bert(input_ids, attention_mask=attention_mask)
#             sequence_output = self.dropout(outputs.last_hidden_state)
#             pooled_output = self.dropout(outputs.pooler_output)

#             ner_logits = self.ner_classifier(sequence_output)
#             intent_logits = self.intent_classifier(pooled_output)

#             return ner_logits, intent_logits
    
#     model = BertForNERAndIntent(model_name, num_labels, num_intents)
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
    
#     return model, tokenizer,label_dict
def load_model_and_tokenizer_entities( device):
    tokenizer = BertTokenizer.from_pretrained('./model/tokenizer')
    num_labels = len(label_dict)
    num_intents = len(intent_dict)

    # Khởi tạo mô hình
    config = BertConfig.from_pretrained('bert-base-uncased')
    config.num_labels = num_labels
    config.num_intents = num_intents
    model = BertForJointIntentAndNER.from_pretrained('bert-base-uncased', config=config, num_intents=num_intents, num_labels=num_labels)

    # Tải trọng số mô hình đã lưu, đảm bảo rằng nó phù hợp với thiết bị
    model.load_state_dict(torch.load('./model/model5.bin', map_location=device))

    # Chuyển mô hình đến thiết bị phù hợp (GPU hoặc CPU)
    model.to(device)

    # Chuyển mô hình sang chế độ đánh giá
    model.eval()
    
    return model, tokenizer,label_dict
def load_chatbot_model(model_path):
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    chatbot = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return chatbot

def load_token_classification_model(model_path, device):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    config = BertConfig.from_pretrained('bert-base-uncased', num_labels=13)
    model = BertForTokenClassification.from_pretrained('bert-base-uncased', config=config)
    model.load_state_dict(torch.load(model_path), strict=False)
    model.to(device)
    return model, tokenizer

def load_image_classification_model(model_path):
    return load_model(model_path)

def load_question_classification_model(model_path):
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return model, tokenizer


def load_json_file_vn(filename):
    with open(filename,'r', encoding='utf-8') as f:
        file = json.load(f)
    return file

def create_df_vn():
    df = pd.DataFrame({
        'Pattern' : [],
        'Tag' : []
    })
    return df
def extract_json_info_vn(json_file, df):
    for intent in json_file['intents']:
        for pattern in intent['patterns']:
            sentence_tag = [pattern, intent['tag']]
            df.loc[len(df.index)] = sentence_tag
    return df
def load_model_vn(model_path, num_labels, id2label, label2id):
    tokenizer = PhobertTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=num_labels, id2label=id2label, label2id=label2id)
    return model, tokenizer

def load_dataset_vn(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data['intents']