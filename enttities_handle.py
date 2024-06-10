from transformers import BertTokenizer, BertForTokenClassification, BertConfig
import torch
import asyncio
import websockets
from functools import partial
import mysql.connector
import json
from decimal import Decimal
from json import JSONEncoder
from decimal import Decimal
# def predict(input_text, model1, tokenizer, device):
#     # Tokenize the input data
#     inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
#     inputs = {key: value.to(device) for key, value in inputs.items()}
#     print(input_text)
#     # Perform prediction
#     with torch.no_grad():
#         outputs = model1(**inputs)
#         predictions = outputs[0]

#     # Convert logits to probabilities and then to labels
#     probabilities = torch.nn.functional.softmax(predictions, dim=-1)
#     predicted_label = probabilities.argmax(dim=-1)
#     return extract_entities(input_text, predicted_label.tolist(),tokenizer)
# def extract_entities(input_text, label_ids,tokenizer):
#     # Tokenize the input text and align labels with tokens
#     tokenized_input = tokenizer.tokenize(input_text)
#     print("Tokens:", tokenized_input)
#     print("Labels:", label_ids)

#     # Dictionary to hold the extracted entities
#     entities = {
#         "category": [],
#         "color": [],
#         "size": [],
#         "gender": [],
#         "price": []
#     }

#     # Helper function to get entity name from label id
#     def get_entity_type(label_id):
#         if label_id == 1:
#             return "category"
#         elif label_id == 3:
#             return "color"
#         elif label_id == 5:
#             return "size"
#         elif label_id == 7:
#             return "gender"
#         elif label_id == 9:
#             return "price"
#         return None
# # Extract entities based on labels
#     for token, label_id in zip(tokenized_input, label_ids[0]):  # assuming label_ids is a 2D list [[...]]
#         entity_type = get_entity_type(label_id)
#         if entity_type:
#             entities[entity_type].append(token)

#     # Convert lists to strings
#     for key in entities:
#         entities[key] = " ".join(entities[key])

#     # Remove empty keys
#     entities = {key: value for key, value in entities.items() if value}
#     print(entities,"hoiiiiii")

#     conditions = " AND ".join([f"{key} = '{value}'" for key, value in entities.items()])
#     res = " , ".join([f"{key} is '{value}'" for key, value in entities.items()])
#     res1= "This is result "+ res

# # SQL query
#     query = f"SELECT * FROM Shoe WHERE {conditions}"
#     print(query)
#     cursor = cnx.cursor(dictionary=True)  # Use dictionary cursor to ease JSON conversion

#     # Execute the query
#     cursor.execute(query)

#     # Fetch all the rows
#     results = cursor.fetchall()

#     # Convert the result into JSON format
#     #json_result = json.dumps(results)
#     json_result = json.dumps(results, cls=DecimalEncoder)
#     return res1+"\n" + json_result
# def predict_with_model(model, tokenizer, label_dict, text,cnx):
#     encoded_input = tokenizer.encode_plus(
#         text, return_tensors='pt', max_length=128, padding='max_length', truncation=True
#     )
#     input_ids = encoded_input['input_ids']
#     attention_mask = encoded_input['attention_mask']

#     with torch.no_grad():
#         ner_logits, intent_logits = model(input_ids, attention_mask)
#         ner_predictions = torch.argmax(ner_logits, dim=-1)
#         intent_predictions = torch.argmax(intent_logits, dim=-1)

#     tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
#     predicted_tags = [list(label_dict.keys())[list(label_dict.values()).index(i)] for i in ner_predictions[0].numpy()]

#     # Filter out 'O' tags and collect entities
#     entities = {}
#     current_entity = None
#     current_entity_name = None

#     for token, tag in zip(tokens, predicted_tags):
#         if tag == 'O':
#             current_entity = None
#             current_entity_name = None
#         elif tag.startswith('B-'):
#             current_entity = token.replace("##", "")
#             current_entity_name = tag[2:]
#             entities[current_entity_name] = current_entity
#         elif tag.startswith('I-') and current_entity_name:
#             if token.startswith("##"):
#                 current_entity += token.replace("##", "")
#             else:
                
#                 current_entity += " " + token
#             entities[current_entity_name] = current_entity
# #     conditions = " AND ".join([f"{key} = '{value}'" for key, value in entities.items()])
# #     res = " , ".join([f"{key} is '{value}'" for key, value in entities.items()])
# #     res1= "This is result "+ res

# # # SQL query
# #     query = f"SELECT * FROM Shoe WHERE {conditions}"
# #     print(query)
# #     cursor = cnx.cursor(dictionary=True)  # Use dictionary cursor to ease JSON conversion

# #     # Execute the query
# #     cursor.execute(query)

# #     # Fetch all the rows
# #     results = cursor.fetchall()
#     prediction_result = {
#         "intent_prediction": intent_predictions.tolist(),
#         "entities": entities
#     }
#     print(prediction_result)
    
#     return json.dumps(prediction_result, indent=4)








intent_dict = {
    'find': 0, 'undefined': 1, 'order_inquiry': 2, 'remove item': 3, 'update cart': 4,
    'buy now': 5, 'cancel order': 6, 'add to cart': 7, 'default': 8
}

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

def preprocess_input(sentence, tokenizer, device, max_length=128):
    encoding = tokenizer(sentence, truncation=True, padding='max_length', max_length=max_length, return_tensors="pt")
    return encoding['input_ids'].to(device), encoding['attention_mask'].to(device),encoding

def predict(sentence, model, tokenizer, device):
    input_ids, attention_mask,encoding = preprocess_input(sentence, tokenizer, device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs['logits']
        intent_logits = outputs['intent_logits']

    entity_predictions = torch.argmax(logits, dim=2).cpu().numpy().flatten()
    intent_predictions = torch.argmax(intent_logits, dim=1).cpu().numpy().flatten()
    print(intent_predictions)
    id2label = {v: k for k, v in label_dict.items()}
    id2intent = {v: k for k, v in intent_dict.items()}

    entity_labels = [id2label[pred] for pred in entity_predictions if pred in id2label]
    intent_label = id2intent[intent_predictions[0]] if intent_predictions[0] in id2intent else 'Unknown'
    sen=tokenizer.convert_ids_to_tokens(encoding['input_ids'].cpu().numpy().flatten())
    entities = {}
    current_entity = ""
    current_entity_name = None
    for token, tag in zip(sen, entity_labels):
        if(token=='[SEP]'):
            break
        if(current_entity_name!=tag[2:] ):
            current_entity = ""
            current_entity_name = ""
        if tag == 'O':
            current_entity = ""
            current_entity_name = ""
        elif tag.startswith('B-'):
            current_entity = token.replace("##", "")
            current_entity_name = tag[2:]
            entities[current_entity_name] = current_entity
        elif tag.startswith('I-'):

            if token.startswith("##"):
                current_entity = current_entity +  token.replace("##", "")
            else:
                if current_entity_name:

                    current_entity = current_entity+" " + token
                else:
                    current_entity_name = tag[2:]
                    current_entity =  token
            entities[current_entity_name] = current_entity
    prediction_result = {
        "intent_prediction": intent_predictions.tolist(),
        "entities": entities
    }
    print(prediction_result)
    
    return json.dumps(prediction_result, indent=4)
    






def add_to_cart_handle(data,cnx):
    missing_info = []
    if not data['entities'].get('name'):
        missing_info.append("name")
    if not data['entities'].get('color'):
        missing_info.append("color")
    if not data['entities'].get('size'):
        missing_info.append("size")
    if missing_info:
        missing_info_message = "Please provide information about " + ", ".join(missing_info)
        result={
            "message":missing_info_message,
            "type":"message"
        }
        return json.dumps(result)
    else:
        name = data['entities'].get('name', '')
        color = data['entities'].get('color', '')
        size = data['entities'].get('size', 0)  # Giá trị mặc định là 0 nếu không tồn tại
        title='AND p1.post_title LIKE' +" '%"+ name+"%'"+' and (p1.post_excerpt like'+" '%Color: "+color+', Size: '+size+"%' or p1.post_excerpt like"+ "'%Size: "+size+', Color: '+color+"%');"
        query = """
  
                SELECT 
                    p1.ID,
                    p1.post_parent,
                    p1.post_excerpt,
                    p2.post_title AS parent_title
                FROM 
                    wp_posts p1
                JOIN 
                    wp_posts p2 ON p1.post_parent = p2.ID
                WHERE 
                    p1.post_type = 'product_variation'
                    

                 """
        query+=title
        print(query)
        cursor = cnx.cursor(dictionary=True)

        try:
            # Execute the query
            cursor.execute(query)
            # Fetch all the rows
            results = cursor.fetchall()
        except Exception as e:
            print("Database error:", e)
            results = []
        print(results)
        #Sử dụng f-string và chuyển đổi size thành chuỗi khi cần ghép nối
        message = f"add to cart success the shoe {name}, {color} color, size {str(size)}"
        result={
            "message":message,
            "type":"add success",
            "data":results
        }
        return json.dumps(result)

# def find_products(data,cnx):
#     res = " , ".join([f"{key} is '{value}'" for key, value in data['entities'].items()])
#     res1= "This is result "+ res
#     # prediction_result = {
#     #     "intent_prediction": intent_predictions.tolist(),
#     #     "entities": entities
#     # }
    
#     # SQL query
#     cursor = cnx.cursor(dictionary=True)
#     # SQL query to fetch product details
#     query = """
#         SELECT
#             p.ID, p.post_title, 

#             pl.sku, pl.min_price, pl.max_price, pl.onsale,
#             pl.stock_quantity, pl.stock_status, pl.rating_count, pl.average_rating, pl.total_sales,
#             pm2.meta_value AS url_img
#         FROM
#             wp_posts p
#         JOIN
#             wp_postmeta pm ON p.ID = pm.post_id
#         JOIN
#             wp_wc_product_meta_lookup pl ON p.ID = pl.product_id
#         LEFT JOIN
#             wp_postmeta pm2 ON pm.meta_value = pm2.post_id AND pm2.meta_key = '_wp_attached_file'
#         WHERE
#            p.post_type = 'product' AND pm.meta_key = '_thumbnail_id' LIMIT 5;"""
#     print(query)
#     cursor = cnx.cursor(dictionary=True)  # Use dictionary cursor to ease JSON conversion

#     # Execute the query
#     cursor.execute(query)

#     # Fetch all the rows
#     results = cursor.fetchall()
#     print(results)
#     result={
#             "message":res1,
#             "type":"product_list",
#             "data":results
#         }
#     print(result)
#     return json.dumps(result, indent=4)
import json
def custom_json_serializer(obj):
    if isinstance(obj, Decimal):
        return float(obj)  # Hoặc str(obj) nếu bạn muốn giữ nguyên định dạng số Decimal nhưng dưới dạng chuỗi
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
def find_products(data, cnx):
    # Generate a descriptive message from entities
    res = ", ".join([f"{key} is '{value}'" for key, value in data['entities'].items() if value is not None])
    message = "This is result " + res
    
    # SQL query to fetch product details
    # query = """
    #     SELECT
    #         p.ID, p.post_title, 
    #         pl.sku, pl.min_price, pl.max_price, pl.onsale,
    #         pl.stock_quantity, pl.stock_status, pl.rating_count, pl.average_rating, pl.total_sales,
    #         pm2.meta_value AS url_img
    #     FROM
    #         wp_posts p
    #     JOIN
    #         wp_postmeta pm ON p.ID = pm.post_id
    #     JOIN
    #         wp_wc_product_meta_lookup pl ON p.ID = pl.product_id
    #     LEFT JOIN
    #         wp_postmeta pm2 ON pm.meta_value = pm2.post_id AND pm2.meta_key = '_wp_attached_file'
    #     WHERE
    #         (p.post_type = 'product' OR p.post_type = 'product_variation') AND pm.meta_key = '_thumbnail_id' LIMIT 5;
    # """
    query = """
    SELECT
        p.ID, p.post_title,
        pl.sku, pl.min_price, pl.max_price, pl.onsale,
        pl.stock_quantity, pl.stock_status, pl.rating_count, pl.average_rating, pl.total_sales,
        pm2.meta_value AS url_img
    FROM
        wp_posts p
    JOIN
        wp_postmeta pm ON p.ID = pm.post_id
    JOIN
        wp_wc_product_meta_lookup pl ON p.ID = pl.product_id
    LEFT JOIN
        wp_postmeta pm2 ON pm.meta_value = pm2.post_id AND pm2.meta_key = '_wp_attached_file'
    WHERE
        (p.post_type = 'product' OR p.post_type = 'product_variation')
        AND pm.meta_key = '_thumbnail_id'
    """

    # Thêm các điều kiện tìm kiếm dựa trên biến
    conditions = []
    for key, value in data['entities'].items():
        if value:
            conditions.append(f"p.post_title LIKE '%{value}%'")
    

    if conditions:
        query += " AND " + " AND ".join(conditions)

    query += " LIMIT 5;"
    print(query)

    # Use dictionary cursor to ease JSON conversion
    cursor = cnx.cursor(dictionary=True)

    try:
        # Execute the query
        cursor.execute(query)
        # Fetch all the rows
        results = cursor.fetchall()
    except Exception as e:
        print("Database error:", e)
        results = []

    result = {
        "message": message,
        "type": "product_list",
        "data": results
    }

    # Output the final result as a formatted JSON string
    json_output = json.dumps(result, default=custom_json_serializer, ensure_ascii=False, indent=4)
    return json_output

def checkoutHandle(data):
    result = {
        "message": "Redirect to the checkout page",
        "type": "redirect",
        "data": "checkout"
    }

    # Output the final result as a formatted JSON string
    json_output = json.dumps(result, default=custom_json_serializer, ensure_ascii=False, indent=4)
    return json_output
    


