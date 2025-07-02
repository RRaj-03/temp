import pandas as pd
import numpy as np
from tqdm import tqdm



!wget -nc https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv

# Step 2: Load CSV using pandas
import pandas as pd
train_df = pd.read_csv("test.csv")

train_df['category_clean'] = train_df['category'].apply(lambda s: s.replace('_', ' '))

label_list = train_df['category_clean'].unique()

print(list(label_list))


# Step 1: Load data
# train_df = pd.read_csv("train.csv")

# Step 2: Define a batched classify function
def batch_classify(texts, top_k=1):
    # GPU-accelerated encoding
    embeddings = model.encode(
        texts,
        batch_size=64,                # Tune this based on your VRAM
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        device='cuda'                # Ensure model is on GPU
    ).astype("float32")

    # FAISS search
    D, I = index.search(embeddings, top_k)

    results = []
    for scores, idxs in zip(D, I):
        preds = []
        for score, idx in zip(scores, idxs):
            leaf = id2rec[leaf_ids[idx]]
            preds.append({
                "score": float(score),
                "leaf_id": leaf["leaf_id"],
                "name": leaf["examples"][0],
                "path": leaf["path"]
            })
        results.append(preds)
    return results

# Step 3: Run batch inference
texts = train_df['text'].tolist()
all_preds = batch_classify(texts, top_k=5)

# Step 4: Save top prediction
train_df['predicted1'] = [pred[0]['name'] if pred else None for pred in all_preds]
train_df['predicted2'] = [pred[1]['name'] if pred else None for pred in all_preds]
train_df['predicted3'] = [pred[2]['name'] if pred else None for pred in all_preds]
train_df['predicted4'] = [pred[3]['name'] if pred else None for pred in all_preds]
train_df['predicted5'] = [pred[4]['name'] if pred else None for pred in all_preds]
train_df['are_equal1'] = train_df['category_clean'] == train_df['predicted1']
train_df['are_equal2'] = train_df['category_clean'] == train_df['predicted2']
train_df['are_equal3'] = train_df['category_clean'] == train_df['predicted3']
train_df['are_equal4'] = train_df['category_clean'] == train_df['predicted4']
train_df['are_equal5'] = train_df['category_clean'] == train_df['predicted5']
sum = 0
accuracy = train_df['are_equal1'].mean()
print(f"Accuracy: {accuracy:.4f}")
sum+=accuracy
accuracy = train_df['are_equal2'].mean()
print(f"Accuracy: {accuracy:.4f}")
sum+=accuracy
accuracy = train_df['are_equal3'].mean()
sum+=accuracy
print(f"Accuracy: {accuracy:.4f}")
accuracy = train_df['are_equal4'].mean()
sum+=accuracy
print(f"Accuracy: {accuracy:.4f}")
accuracy = train_df['are_equal5'].mean()
sum+=accuracy
print(f"Accuracy: {accuracy:.4f}")
print(sum)
# Step 5: Save to CSV
train_df.to_csv("train_with_predictions_batch1.csv", index=False)

# Optional preview
print(train_df[['text', 'predicted1']].head())



import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
# Step 1: Load data
# train_df = pd.read_csv("train.csv")
!wget -nc https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv

# Step 2: Load CSV using pandas
import pandas as pd
train_df = pd.read_csv("test.csv")

train_df['category_clean'] = train_df['category'].apply(lambda s: s.replace('_', ' '))

label_list = train_df['category_clean'].unique()

print(list(label_list))
# ================================================================
# 1.  SAMPLE DATA  (swap in your 750 phrases)
# ================================================================
phrases = list(label_list)

# Step 2: Define a batched classify function
def batch_classify(texts, top_k=1):
    # GPU-accelerated encoding
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
    model = AutoModelForSequenceClassification.from_pretrained("Qwen/Qwen2.5-3B")  # or use your own fine-tuned model

    # Define classes (phrases)
    # phrases = ["finance", "account", "fraud", "payment", "insurance"]
    # texts = "The transaction was flagged due to suspicious activity."

    # Create prompt (optional)
    text = f"You have {len(phrases)} classes: {', '.join(phrases)}. Classify: {texts}"

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        predicted_class_idx = torch.argmax(probs, dim=1).item()

    # embeddings = model.encode(
    #     texts,
    #     batch_size=64,                # Tune this based on your VRAM
    #     convert_to_numpy=True,
    #     normalize_embeddings=True,
    #     show_progress_bar=True,
    #     device='cuda'                # Ensure model is on GPU
    # ).astype("float32")

    # FAISS search
    # D, I = index.search(embeddings, top_k)

    # results = []
    # for scores, idxs in zip(D, I):
    #     preds = []
    #     for score, idx in zip(scores, idxs):
    #         leaf = id2rec[leaf_ids[idx]]
    #         preds.append({
    #             "score": float(score),
    #             "leaf_id": leaf["leaf_id"],
    #             "name": leaf["examples"][0],
    #             "path": leaf["path"]
    #         })
    #     results.append(preds)
    return  phrases[predicted_class_idx]

# Step 3: Run batch inference
train_df = train_df[:250]
texts = train_df['text'].tolist()

all_preds =[]
for text in texts:
  a = batch_classify(text, top_k=35)
  all_preds.append(a)

# Step 4: Save top prediction
train_df['predicted1'] = [pred if pred else None for pred in all_preds]
# train_df['predicted2'] = [pred[1]['name'] if pred else None for pred in all_preds]
# train_df['predicted3'] = [pred[2]['name'] if pred else None for pred in all_preds]
# train_df['predicted4'] = [pred[3]['name'] if pred else None for pred in all_preds]
# train_df['predicted5'] = [pred[4]['name'] if pred else None for pred in all_preds]
train_df['are_equal1'] = train_df['category_clean'] == train_df['predicted1']
# train_df['are_equal2'] = train_df['category_clean'] == train_df['predicted2']
# train_df['are_equal3'] = train_df['category_clean'] == train_df['predicted3']
# train_df['are_equal4'] = train_df['category_clean'] == train_df['predicted4']
# train_df['are_equal5'] = train_df['category_clean'] == train_df['predicted5']
sum = 0
accuracy = train_df['are_equal1'].mean()
print(f"Accuracy: {accuracy:.4f}")
sum+=accuracy
# accuracy = train_df['are_equal2'].mean()
# print(f"Accuracy: {accuracy:.4f}")
# sum+=accuracy
# accuracy = train_df['are_equal3'].mean()
# sum+=accuracy
# print(f"Accuracy: {accuracy:.4f}")
# accuracy = train_df['are_equal4'].mean()
# sum+=accuracy
# print(f"Accuracy: {accuracy:.4f}")
# accuracy = train_df['are_equal5'].mean()
# sum+=accuracy
# print(f"Accuracy: {accuracy:.4f}")
print(sum)
# Step 5: Save to CSV
train_df.to_csv("train_with_predictions_batch2.csv", index=False)

# Optional preview
print(train_df[['text', 'predicted1']].head())
