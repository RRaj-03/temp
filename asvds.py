# %%
from datasets import load_dataset

try:
    # 1. Load the dataset
    print("Loading PolyAI/banking77 dataset...")
    dataset = load_dataset("PolyAI/banking77")
    print("Dataset loaded successfully.")
    print(dataset)

    # 2. Get the list of label names (strings) - this is crucial for mapping
    # We'll get it from the 'train' split, but it's consistent across splits.
    label_names = dataset['train'].features['label'].names
    print(f"\nLabel names found: {label_names[:5]}... (showing first 5)") # Show a few to confirm

    # 3. Define a mapping function to create a new column with string labels
    def create_string_label_column(example):
        """
        Maps the numerical 'label' ID to its corresponding string name
        and assigns it to a new 'label_string' column.
        """
        # Ensure 'label' exists and is an integer
        if 'label' in example and isinstance(example['label'], int):
            example['label_string'] = label_names[example['label']]
        else:
            # Handle cases where 'label' might be missing or not an int
            example['label_string'] = None # Or some other default value
        return example

    # 4. Apply the mapping function to both 'train' and 'test' splits
    print("\nMapping numerical labels to string labels...")
    dataset = dataset.map(create_string_label_column)
    print("Mapping complete.")

    # 5. Drop the old numerical 'label' column and rename 'label_string' to 'label'
    print("\nRestructuring dataset columns...")
    for split in dataset:
        if 'label' in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns(['label'])
        if 'label_string' in dataset[split].column_names:
            dataset[split] = dataset[split].rename_column('label_string', 'label')
    print("Dataset restructuring complete.")

    # 6. Verify the change
    print("\nFinal Dataset Structure:")
    print(dataset)
    print("\nFirst example from train split after conversion:")
    print(dataset['train'][0])

    print("\nFirst example from test split after conversion:")
    print(dataset['test'][0])

    # You can also verify the features structure again.
    print("\nNew features structure for train split:")
    print(dataset['train'].features)

except Exception as e:
    print(f"\nAn error occurred: {e}")
    import traceback
    traceback.print_exc()

# %%













print(list(label_names))

# %%
# ================================================================
# COMPLETE PIPELINE: sparse clustering + metrics + cluster naming
# ================================================================
import importlib, subprocess, sys, re, time, math
from collections import Counter, defaultdict

# --- install deps silently if missing (Colab-friendly) -------------
for pkg in ["numpy", "scipy", "matplotlib", "scikit-learn"]:
    if importlib.util.find_spec(pkg) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

# --- std lib / scientific ------------------------------------------
import numpy as np
from scipy.sparse             import csr_matrix
from scipy.cluster.hierarchy  import linkage, dendrogram, cophenet, fcluster
from scipy.spatial.distance   import squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics          import silhouette_score
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from  tqdm import tqdm 

!wget -nc https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv

# Step 2: Load CSV using pandas
import pandas as pd
train_df = pd.DataFrame(dataset['test'])

train_df['category_clean'] = train_df['label'].apply(lambda s: s.replace('_', ' '))

label_list = train_df['category_clean'].unique()

print(list(label_list))

# ================================================================
# 1.  SAMPLE DATA  (swap in your 750 phrases)
# ================================================================
phrases = list(label_list)

# ================================================================
# 2.  TOKENISE → SPARSE PROBABILITY MATRIX
# ================================================================
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# tok = re.compile(r"\b\w+\b").findall
tokenised = []
for p in tqdm(phrases, desc="Tokenizing phrases"):
    # Encode the phrase to get token IDs
    ids = tok.encode(p.lower())
    # Convert token IDs to string tokens
    string_tokens = tok.convert_ids_to_tokens(ids)
    tokenised.append(string_tokens)
print(tokenised)
vocab  = sorted({t for s in tokenised for t in tqdm(s,desc="vocab")})
v2i    = {w: i for i, w in enumerate(vocab)}
rows, cols, data = [], [], []
for r, sent in tqdm(enumerate(tokenised)):
    cnt = Counter(sent);  tot = sum(cnt.values())
    for w, c in cnt.items():
        rows.append(r); cols.append(v2i[w]); data.append(c / tot)
X = csr_matrix((data, (rows, cols)), shape=(len(phrases), len(vocab)))

# ================================================================
# 3.  GLOBAL STATS (df, idf, corpus background)
# ================================================================
N        = len(phrases)
df       = Counter(t for s in tokenised for t in set(s))
idf_vec  = np.array([math.log(N / df[w]) for w in vocab])
idf_rev  = {i: idf_vec[i] for i in range(len(vocab))}

bg_cnts  = Counter(t for s in tokenised for t in s)
bg_total = sum(bg_cnts.values())
bg       = np.array([bg_cnts.get(w, 0)/bg_total for w in vocab])
beta_vec = 0.2 * idf_vec / idf_vec.max()          # adaptive β
eps = 1e-9

# ================================================================
# 4.  THREE DIVERGENCES
# ================================================================
def wkls(u, v):
    val = 0.0
    for idx, pu in zip(u.indices, u.data):
        pv = v[0, idx]; w = idf_rev[idx]
        val += w * pu * math.log((pu+eps)/(pv+eps))
    for idx, pv in zip(v.indices, v.data):
        pu = u[0, idx]; w = idf_rev[idx]
        val += w * pv * math.log((pv+eps)/(pu+eps))
    return val

def ap_jsd(u, v):
    p = u.toarray().ravel();  q = v.toarray().ravel()
    p_ = (1-beta_vec)*p + beta_vec*bg
    q_ = (1-beta_vec)*q + beta_vec*bg
    m  = 0.5*(p_+q_)
    js = 0.5*np.sum(p_*np.log((p_+eps)/(m+eps))) + \
         0.5*np.sum(q_*np.log((q_+eps)/(m+eps)))
    return js

def renyi_jsd(u, v, alpha=0.8):
    p = u.toarray().ravel() + eps
    q = v.toarray().ravel() + eps
    m = 0.5*(p+q)
    num   = (m**alpha).sum()
    denom = 0.5*(p**alpha).sum() + 0.5*(q**alpha).sum()
    val   = (math.log(num) - math.log(denom)) / (alpha-1)
    return max(val, 0.0)

divergences = {
    "WKLS"   : wkls,
    "AP-JSD" : ap_jsd,
    #"Rényi-JSD": renyi_jsd,
}

# ================================================================
# 5.  BUILD CONDENSED DISTANCE  (shared helper)
# ================================================================
def condensed(div):
    n = X.shape[0]; out = np.empty(n*(n-1)//2)
    k = 0
    for i in range(n-1):
        ui = X.getrow(i)
        for j in range(i+1, n):
            d = div(ui, X.getrow(j))
            out[k] = d if d >= 0 else 0.0
            k += 1
    return out

# ================================================================
# 6.  DRAW DENDROGRAMS & COLLECT METRICS
# ================================================================
k_cut = 5
metrics = []
for name, f in divergences.items():
    D = condensed(f)
    Z = linkage(D, method="complete")

    # --- intrinsic metrics ------------------
    ccc, _ = cophenet(Z, D)
    labels_k = fcluster(Z, k_cut, criterion="maxclust")
    sil = silhouette_score(squareform(D), labels_k, metric="precomputed")
    metrics.append((name, ccc, sil, Z, labels_k, D))

    # --- plot dendrogram (comment out if running headless) ----
    plt.figure(figsize=(10,4))
    dendrogram(Z, labels=phrases, leaf_rotation=90, leaf_font_size=8)
    plt.title(f"{name} dendrogram")
    plt.tight_layout(); plt.show()

print(f"\n{'Divergence':12s}  CCC   Sil@{k_cut}")
for n, c, s, *_ in metrics:
    print(f"{n:12s}  {c:5.3f}  {s:5.3f}")

# pick the divergence with highest silhouette
best = max(metrics, key=lambda t: t[2])   # (name, ccc, sil, Z, labels, D)
print(f"\n>> Using {best[0]} for naming.\n")

best_Z, best_labels = best[3], best[4]

# ================================================================
# 7.  SIMPLE CLUSTER NAMING  (TF-IDF + heuristic)
# ================================================================
def top_ngrams(member_ids, n_top=4):
    texts = [phrases[i] for i in member_ids]
    tfidf = TfidfVectorizer(ngram_range=(1,2), stop_words='english',
                            min_df=1).fit(texts)
    scores = np.asarray(tfidf.transform(texts).sum(axis=0)).ravel()
    best_idx = scores.argsort()[::-1][:n_top]
    return [tfidf.get_feature_names_out()[i] for i in best_idx]

def heuristic_label(cands):
    # pick the shortest candidate, title-case it
    return min(cands, key=len).title()

clusters = defaultdict(list)
for idx, cid in enumerate(best_labels):
    clusters[cid].append(idx)

cluster_names = {}
for cid, members in clusters.items():
    cands = top_ngrams(members)
    cluster_names[cid] = heuristic_label(cands)

# ================================================================
# 8.  DISPLAY RESULTING CLUSTERS + NAMES
# ================================================================
for cid, members in sorted(clusters.items()):
    print(f"\n▣  Cluster {cid} — {cluster_names[cid]}")
    for idx in members:
        print("   •", phrases[idx])


# %%
# ================================================================
# 1-CELL PIPELINE  —  clustering • metrics • HF zero-shot naming
# ================================================================
import importlib, subprocess, sys, re, math, json, os
from collections import Counter, defaultdict

# ── install lightweight deps silently ────────────────────────────
for pkg in ("numpy", "scipy", "matplotlib", "scikit-learn",
            "transformers", "sentencepiece", "accelerate"):
    if importlib.util.find_spec(pkg) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

# ── std + scientific imports ─────────────────────────────────────
import numpy as np
from scipy.sparse             import csr_matrix
from scipy.cluster.hierarchy  import linkage, cophenet, fcluster
from scipy.spatial.distance   import squareform
from sklearn.metrics          import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt                                 # comment if headless
from transformers import pipeline, AutoTokenizer                              # HF zero-shot naming

# ================================================================
# DATA  (swap in your 750 phrases)
# ================================================================
phrases = list(label_list)
print(phrases)
# ================================================================
# TOKENISE → sparse probability matrix
# ================================================================
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# tok = re.compile(r"\b\w+\b").findall
tokenised = []
for p in tqdm(phrases, desc="Tokenizing phrases"):
    # Encode the phrase to get token IDs
    ids = tok.encode(p.lower())
    # Convert token IDs to string tokens
    string_tokens = tok.convert_ids_to_tokens(ids)
    tokenised.append(string_tokens)
print(tokenised)
vocab  = sorted({t for s in tokenised for t in tqdm(s,desc="vocab")})
v2i    = {w: i for i, w in enumerate(vocab)}
rows, cols, data = [], [], []
for r, sent in tqdm(enumerate(tokenised)):
    cnt = Counter(sent);  tot = sum(cnt.values())
    for w, c in cnt.items():
        rows.append(r); cols.append(v2i[w]); data.append(c / tot)
X = csr_matrix((data, (rows, cols)), shape=(len(phrases), len(vocab)))

# ================================================================
# global stats for divergences
# ================================================================
N = len(phrases)
df = Counter(tok for sent in tokenised for tok in set(sent))
idf_vec = np.array([math.log(N / df[w]) for w in vocab])
idf_map = {i: idf_vec[i] for i in range(len(vocab))}

bg_counts = Counter(tok for sent in tokenised for tok in sent)
bg_total  = sum(bg_counts.values())
bg        = np.array([bg_counts.get(w, 0)/bg_total for w in vocab])
beta_vec  = 0.2 * idf_vec / idf_vec.max()
eps = 1e-9

# ================================================================
# divergence definitions
# ================================================================
def wkls(u, v):
    out = 0.0
    for idx, pu in zip(u.indices, u.data):
        pv = v[0, idx]; w = idf_map[idx]
        out += w * pu * math.log((pu+eps)/(pv+eps))
    for idx, pv in zip(v.indices, v.data):
        pu = u[0, idx]; w = idf_map[idx]
        out += w * pv * math.log((pv+eps)/(pu+eps))
    return out

def ap_jsd(u, v):
    p = u.toarray().ravel(); q = v.toarray().ravel()
    p_ = (1-beta_vec)*p + beta_vec*bg
    q_ = (1-beta_vec)*q + beta_vec*bg
    m  = 0.5*(p_+q_)
    return 0.5*np.sum(p_*np.log((p_+eps)/(m+eps))) + \
           0.5*np.sum(q_*np.log((q_+eps)/(m+eps)))

def renyi_jsd(u, v, a=0.8):
    p = u.toarray().ravel()+eps; q = v.toarray().ravel()+eps
    m = 0.5*(p+q)
    num   = (m**a).sum()
    denom = 0.5*(p**a).sum() + 0.5*(q**a).sum()
    return max((math.log(num)-math.log(denom))/(a-1), 0.0)

divergences = {"WKLS": wkls, "AP-JSD": ap_jsd, #"Rényi-JSD": renyi_jsd
               }

def condensed(div):
    n = X.shape[0]; out = np.empty(n*(n-1)//2)
    k = 0
    for i in range(n-1):
        ui = X.getrow(i)
        for j in range(i+1, n):
            out[k] = div(ui, X.getrow(j)); k += 1
    return out

# ================================================================
# choose best divergence by silhouette (k=5)
# ================================================================
k_eval = 5
best_name, best_Z, best_D = None, None, None
best_sil = -1.0
for name, f in divergences.items():
    D = condensed(f);  Z = linkage(D, method="complete")
    sil = silhouette_score(squareform(D),
                           fcluster(Z, k_eval, criterion="maxclust"),
                           metric="precomputed")
    print(f"{name:8s} silhouette@{k_eval}: {sil:5.3f}")
    if sil > best_sil:
        best_name, best_Z, best_D, best_sil = name, Z, D, sil
print(f"\n>> chosen divergence: {best_name}\n")

# ================================================================
# build parent/child dictionaries from linkage
# ================================================================
n = len(phrases)
parents  = {}
children = defaultdict(list)
for idx, (a, b, _, _) in enumerate(best_Z, start=n):
    parents[int(a)] = parents[int(b)] = idx
    children[idx].extend([int(a), int(b)])

# ================================================================
# Hugging-Face zero-shot naming model
# ================================================================
hf_gen = pipeline("text2text-generation",
                  model="google/flan-t5-small",
                  max_length=8,
                  do_sample=False)

def zero_shot_name(cands):
    prompt = ("Give a concise 3-word category title for: " +
              "; ".join(cands))
    text = hf_gen(prompt, num_return_sequences=1)[0]["generated_text"]
    # take first ≤4 words
    return " ".join(text.split()[:4]).title()

def tfidf_terms(member_ids, k=6):
    texts = [phrases[i] for i in member_ids]
    vec   = TfidfVectorizer(ngram_range=(1,2), stop_words="english").fit(texts)
    scores = np.asarray(vec.transform(texts).sum(axis=0)).ravel()
    idxs   = scores.argsort()[::-1][:k]
    return [vec.get_feature_names_out()[i] for i in idxs]

# ================================================================
# build node representations bottom-up
# ================================================================
node = {i: {"id": i, "phrase": phrases[i]} for i in range(n)}

# internal nodes
for idx in range(n, n+len(best_Z)):
    # gather all descendant leaves
    stack = [idx]; leaves = []
    while stack:
        v = stack.pop()
        if v < n: leaves.append(v)
        else:     stack.extend(children[v])
    cands = tfidf_terms(leaves)
    node[idx] = {
        "id": idx,
        "name": zero_shot_name(cands),
        "children": [node[c] for c in children[idx]]
    }

root = node[n+len(best_Z)-1]

# ================================================================
# save + preview JSON
# ================================================================
with open("hierarchy.json", "w") as f:
    json.dump(root, f, indent=2)
print(json.dumps(root, indent=2)[:1200], "\n...\n(JSON truncated)")
print("\nHierarchy saved to hierarchy.json")


# %%
!pip install -q sentence-transformers transformers accelerate faiss-cpu

# %%
import json, pathlib, itertools, faiss, numpy as np
from sentence_transformers import SentenceTransformer

json_tree = json.load(open("hierarchy.json"))
model = SentenceTransformer("Qwen/Qwen3-Embedding-4B")

leaf_records = []

def collect(node, path):
    if "phrase" in node:          # leaf
        leaf_records.append({"leaf_id": node["id"],
                             "path": " ▸ ".join(path),
                             "examples": [node["phrase"]]})
    else:
        for ch in node["children"]:
            collect(ch, path+[node["name"]])
collect(json_tree, [json_tree["name"]])


# %%
embeds, leaf_ids = [], []
for rec in leaf_records:
    vec = model.encode(rec["examples"], normalize_embeddings=True).mean(axis=0)
    embeds.append(vec)
    leaf_ids.append(rec["leaf_id"])

embeds = np.stack(embeds).astype("float32")
index  = faiss.IndexFlatIP(embeds.shape[1])        # cosine (after L2-norm)
index.add(embeds)
id2rec = {rec["leaf_id"]: rec for rec in leaf_records}


# %%
def classify(text, top_k=3):
    q = model.encode(text, normalize_embeddings=True).astype("float32")
    D, I = index.search(q.reshape(1,-1), top_k)    # cosine scores
    results = []
    for score, idx in zip(D[0], I[0]):
        leaf = id2rec[leaf_ids[idx]]
        results.append({"score": float(score),
                        "leaf_id": leaf["leaf_id"],
                        "name":leaf["examples"][0],
                        "path": leaf["path"]})
    return results

test = "can I move my 401k over to fidelity?"
print(classify(test)[0])      # ⇒ likely “Account ▸ 401K ▸ Transfer 401K”


# %%
def classify(text, top_k=3):
    q = model.encode(text, normalize_embeddings=True).astype("float32")
    D, I = index.search(q.reshape(1,-1), top_k)    # cosine scores
    results = []
    for score, idx in zip(D[0], I[0]):
        leaf = id2rec[leaf_ids[idx]]
        results.append({"score": float(score),
                        "leaf_id": leaf["leaf_id"],
                        "path": leaf["path"]})
    return results

test = "i want to trade in mutual funds"
print(classify(test)[0])     


# %%
import pandas as pd
import numpy as np
from tqdm import tqdm

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
train_df.to_csv("train_with_predictions_batch11.csv", index=False)

# Optional preview
print(train_df[['text', 'predicted1']].head())



# %%
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM # Import AutoModelForCausalLM

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
BATCH_SIZE = 16 # Adjust based on your GPU memory. Causal LMs can be more memory intensive.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Step 1: Load data ---
# Ensure you have the train.csv file. If not, uncomment the wget command.
# !wget -nc https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv
train_df = pd.read_csv("train.csv") # Assuming train.csv is present

train_df['category_clean'] = train_df['label'].apply(lambda s: s.replace('_', ' '))
label_list = train_df['category_clean'].unique()
print("Available labels:", list(label_list))

# --- Step 2: Load model and tokenizer once ---
print(f"Loading tokenizer and model from {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Add a padding token if it's not already defined (common for causal LMs when batching)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token # Or any other suitable token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval() # Set model to evaluation mode

# --- 1. SAMPLE DATA (your phrases/labels) ---
phrases = list(label_list)
num_phrases = len(phrases)
print(f"Number of phrases/classes: {num_phrases}")

# --- Helper Function for Causal LM Classification ---
def classify_with_causal_lm_batch(texts_batch, phrases, tokenizer, model, device):
    predicted_labels = []
    
    for text in texts_batch:
        # Craft a prompt that encourages the model to generate one of your labels
        # Experiment with different prompt formats for best results!
        # Option 1: Simple completion
        # prompt = f"Text: \"{text}\"\nThis text is about "

        # Option 2: More explicit instruction
        # prompt = f"Classify the following text into one of these categories: {', '.join(phrases)}.\nText: \"{text}\"\nCategory: "

        # Option 3: Question-answering style
        # prompt = f"Given the text: \"{text}\"\nWhat is the most relevant category from this list: {', '.join(phrases)}?\nAnswer: "
        
        # Using a balanced prompt for better performance with Qwen
        prompt = f"Given the text: \"{text}\"\nClassify it into one of these categories: {', '.join(phrases)}.\nCategory:"


        # Tokenize the prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Generate completions for each possible label
        # We'll calculate the likelihood of each phrase being the *next* token(s)
        # after the prompt, or by having the model generate a short sequence
        # and then checking if it matches one of our labels.

        # --- Method 1: Score the likelihood of generating each phrase (more robust) ---
        # This involves calculating the log-probability of each phrase given the prompt.
        # This is more accurate but computationally more intensive as you run inference
        # for each phrase.

        scores = {}
        for phrase in phrases:
            # Construct a full sequence: prompt + phrase
            full_prompt = prompt + " " + phrase # Add a space for tokenization
            full_input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(device)

            # Get logits from the model
            with torch.no_grad():
                outputs = model(full_input_ids)
                logits = outputs.logits

            # Calculate the log-probability of generating the 'phrase' tokens
            # We need to consider only the logits corresponding to the `phrase` part
            # This requires careful indexing.
            
            # Find the start index of the phrase tokens
            prompt_len = input_ids.shape[1]
            phrase_token_ids = full_input_ids[:, prompt_len:]

            # If phrase_token_ids is empty, skip or handle error
            if phrase_token_ids.numel() == 0:
                scores[phrase] = -float('inf') # Assign very low score
                continue

            # Logits for the phrase tokens (shifted for next token prediction)
            # The model predicts the next token based on the current sequence.
            # So, logits[0, prompt_len-1] is for the first token of the phrase,
            # logits[0, prompt_len] is for the second, etc.
            
            # The log_softmax is applied over the vocabulary dimension
            log_probs = F.log_softmax(logits[0, prompt_len - 1 : -1], dim=-1)

            # Gather the log probabilities of the actual phrase tokens
            # Sum the log probabilities to get the score for the entire phrase
            phrase_log_prob = 0.0
            for i, token_id in enumerate(phrase_token_ids[0]):
                if i < log_probs.shape[0]: # Ensure we don't go out of bounds
                    phrase_log_prob += log_probs[i, token_id].item()
                else:
                    # This case should ideally not happen if prompt_len-1:-1 covers all phrase tokens
                    # but safety check
                    break
            
            scores[phrase] = phrase_log_prob

        if not scores: # Handle cases where no scores were calculated
            predicted_labels.append("UNKNOWN")
            continue

        # Select the phrase with the highest score
        best_phrase = max(scores, key=scores.get)
        predicted_labels.append(best_phrase)

    return predicted_labels

# --- Step 3: Run batch inference ---
train_df = train_df[:1000] # For testing, reduce size if needed
texts = train_df['text'].tolist()

all_preds = []
print(f"Processing {len(texts)} texts in batches of {BATCH_SIZE}...")

# Process in smaller chunks to avoid memory issues when encoding the prompt + each phrase
# This loop is for text batches, but inside `classify_with_causal_lm_batch`, we iterate through phrases.
# So, it's effectively (num_texts / BATCH_SIZE) * num_phrases * num_tokens_in_phrase inferences.
# This can still be slow if num_phrases is large.

# For truly large label lists, using embedding similarity (e.g., Sentence-BERT) is often faster
# than NLL-based scoring with a Causal LM, as it's one embedding per text and one per label.
# NLI-based (HuggingFace pipeline) is also highly optimized.

for i in tqdm(range(0, len(texts), BATCH_SIZE)):
    batch_texts = texts[i:i + BATCH_SIZE]
    # The batching for `classify_with_causal_lm_batch` is now handled internally,
    # but the outer loop still serves to chunk the main `texts` list.
    preds = classify_with_causal_lm_batch(batch_texts, phrases, tokenizer, model, DEVICE)
    all_preds.extend(preds)

# --- Step 4: Save top prediction ---
train_df['predicted1'] = all_preds
train_df['are_equal1'] = train_df['category_clean'] == train_df['predicted1']
accuracy = train_df['are_equal1'].mean()
print(f"Accuracy: {accuracy:.4f}")

train_df.to_csv("train_with_predictions_causallm.csv", index=False)

# Optional preview
print(train_df[['text', 'predicted1', 'category_clean', 'are_equal1']].head())

# %%
import pandas as pd # Assuming you'll use pandas for the main script
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from tqdm.auto import tqdm # For progress bar

# --- Helper Function for Causal LM Classification (as provided by you) ---
def classify_with_causal_lm_batch(texts_batch, phrases, tokenizer, model, device):
    predicted_labels = []

    for text in texts_batch:
        prompt = f"Given the text: \"{text}\"\nClassify it into one of these categories: {', '.join(phrases)}.\nCategory:"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        scores = {}
        for phrase in phrases:
            full_prompt = prompt + " " + phrase
            full_inputs = tokenizer(full_prompt, return_tensors="pt").to(device) # Tokenize with attention_mask

            with torch.no_grad():
                outputs = model(**full_inputs) # Correctly passes input_ids and attention_mask
                logits = outputs.logits

            # IMPORTANT: Ensure `inputs` and `full_inputs` have consistent tokenization settings (e.g., add_special_tokens)
            # when determining prompt_len and phrase_token_ids.
            # Using encode for full_input_ids and tokenizer() for inputs (which gets attention_mask automatically)
            # is fine as long as they are consistent in how special tokens are handled.
            
            # Recalculate prompt_len from the original `inputs` for consistency
            prompt_len = inputs.input_ids.shape[1] 
            
            # Make sure `phrase_token_ids` comes from the `full_inputs` and starts *after* the base prompt
            # The [0] is for the batch dimension, [:, prompt_len:] takes tokens from that point onwards
            phrase_token_ids = full_inputs.input_ids[:, prompt_len:]

            if phrase_token_ids.numel() == 0:
                scores[phrase] = -float('inf')
                continue

            # Ensure logits indexing is correct for the start of the phrase tokens
            # The logits are for predicting the *next* token. So, to get the log_prob
            # of the first phrase token, we look at logits at index `prompt_len - 1`.
            # If the sequence is [..., token_A, token_B, token_C]
            # logits[..., token_A_index] predict token_B
            # logits[..., token_B_index] predict token_C
            # So, for the first token of the phrase, we need logits corresponding to the token *just before* it.
            log_probs = F.log_softmax(logits[0, prompt_len - 1 : -1], dim=-1)

            phrase_log_prob = 0.0
            for i, token_id in enumerate(phrase_token_ids[0]): # Iterate over tokens in the phrase
                if i < log_probs.shape[0]: # Check if current index is within log_probs dimension
                    phrase_log_prob += log_probs[i, token_id].item()
                else:
                    # This means the phrase is longer than the generated logits slice.
                    # Should not happen if `max_new_tokens` (if generating) or `full_prompt` length
                    # is handled correctly for getting logits.
                    break
            
            scores[phrase] = phrase_log_prob

        if not scores:
            predicted_labels.append("UNKNOWN")
            continue

        best_phrase = max(scores, key=scores.get)
        predicted_labels.append(best_phrase)

    return predicted_labels

# --- Global Configurations (moved from previous main script) ---
BATCH_SIZE = 16 # Adjust based on your GPU memory. Causal LMs can be more memory intensive.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Example Usage ---

# Define your classes
train_df = pd.DataFrame(dataset['train'])
train_df['category_clean'] = train_df['label'].apply(lambda s: s.replace('_', ' '))

label_list = train_df['category_clean'].unique()
phrases = list(label_list)
# Example texts
single_text = "My credit card was charged multiple times for a single transaction. I suspect unauthorized activity."
train_df = train_df[:100]
texts = train_df['text'].tolist()

batch_texts = texts

print(f"--- Classifying using Qwen2.5-0.5B ---")
# Load model and tokenizer for Qwen2.5-0.5B
MODEL_NAME_QWEN2_5 = "Qwen/Qwen2.5-0.5B"
print(f"Loading tokenizer and model from {MODEL_NAME_QWEN2_5}...")
tokenizer_qwen2_5 = AutoTokenizer.from_pretrained(MODEL_NAME_QWEN2_5)
if tokenizer_qwen2_5.pad_token is None:
    tokenizer_qwen2_5.pad_token = tokenizer_qwen2_5.eos_token
model_qwen2_5 = AutoModelForCausalLM.from_pretrained(MODEL_NAME_QWEN2_5)
model_qwen2_5.to(DEVICE)
model_qwen2_5.eval()

# Classify a single text (wrap in a list for the batch function)
predicted_single = classify_with_causal_lm_batch([single_text], phrases, tokenizer_qwen2_5, model_qwen2_5, DEVICE)[0]
print(f"\nText: \"{single_text}\"\nPredicted Class: {predicted_single}")

# Classify a batch of texts
predicted_batch = classify_with_causal_lm_batch(batch_texts, phrases, tokenizer_qwen2_5, model_qwen2_5, DEVICE)
print(f"\nBatch Classification Results (Qwen2.5-0.5B):")
for text, pred_class in zip(batch_texts, predicted_batch):
    print(f"  Text: \"{text[:50]}...\"\n  Predicted Class: {pred_class}")
train_df['predicted1'] = predicted_batch
train_df['are_equal1'] = train_df['category_clean'] == train_df['predicted1']
# Clear memory if running multiple models
del model_qwen2_5
del tokenizer_qwen2_5
torch.cuda.empty_cache()


print(f"\n--- Classifying using Qwen3-0.5B ---")
try:
    MODEL_NAME_QWEN3 = "Qwen/Qwen3-0.5B"
    print(f"Loading tokenizer and model from {MODEL_NAME_QWEN3}...")
    tokenizer_qwen3 = AutoTokenizer.from_pretrained(MODEL_NAME_QWEN3)
    if tokenizer_qwen3.pad_token is None:
        tokenizer_qwen3.pad_token = tokenizer_qwen3.eos_token
    model_qwen3 = AutoModelForCausalLM.from_pretrained(MODEL_NAME_QWEN3)
    model_qwen3.to(DEVICE)
    model_qwen3.eval()

    predicted_single_qwen3 = classify_with_causal_lm_batch([single_text], phrases, tokenizer_qwen3, model_qwen3, DEVICE)[0]
    print(f"\nText: \"{single_text}\"\nPredicted Class: {predicted_single_qwen3}")

    predicted_batch_qwen3 = classify_with_causal_lm_batch(batch_texts, phrases, tokenizer_qwen3, model_qwen3, DEVICE)
    print(f"\nBatch Classification Results (Qwen3-0.5B):")
    for text, pred_class in zip(batch_texts, predicted_batch_qwen3):
        print(f"  Text: \"{text[:50]}...\"\n  Predicted Class: {pred_class}")
    train_df['predicted2'] = predicted_batch
    train_df['are_equal2'] = train_df['category_clean'] == train_df['predicted2']
    train_df.to_csv("a.csv")

except Exception as e:
    print(f"\nCould not load or run Qwen3-0.5B. Ensure it's correctly installed/available. Error: {e}")

# If you were running the full script with train_df, remember to include that part.
# The `train_df` related code would go after the model loading for the specific Qwen version you want to use.
# For example:
# train_df = pd.read_csv("train.csv")
# train_df['category_clean'] = train_df['label'].apply(lambda s: s.replace('_', ' '))
# label_list = train_df['category_clean'].unique()
# phrases_from_df = list(label_list)
# texts_from_df = train_df['text'].tolist()
# all_preds_df = []
# for i in tqdm(range(0, len(texts_from_df), BATCH_SIZE)):
#     batch_texts_df = texts_from_df[i:i + BATCH_SIZE]
#     preds_df = classify_with_causal_lm_batch(batch_texts_df, phrases_from_df, tokenizer_qwen2_5, model_qwen2_5, DEVICE)
#     all_preds_df.extend(preds_df)
# # ... (accuracy calculation and saving)

# %%
train_df.to_csv("ab1.csv",index=False)

# %%

import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
# Step 1: Load data
# train_df = pd.read_csv("train.csv")
# !wget -nc https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv

# Step 2: Load CSV using pandas
import pandas as pd
# train_df = pd.read_csv("test.csv")

train_df['category_clean'] = train_df['label'].apply(lambda s: s.replace('_', ' '))

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


