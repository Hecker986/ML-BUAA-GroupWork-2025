import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda")
MODEL = "hfl/chinese-roberta-wwm-ext-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

test_df = pd.read_pickle("test_no_label.pkl")
X_test = test_df["text"].values

def tokenize(texts):
    return tokenizer(list(texts), truncation=True, max_length=512,
                     padding="max_length", return_tensors="pt")

enc = tokenize(X_test)

class DS(torch.utils.data.Dataset):
    def __init__(self, enc):
        self.enc = enc
    def __len__(self): return len(self.enc['input_ids'])
    def __getitem__(self,i):
        return {k:v[i] for k,v in self.enc.items()}

loader = torch.utils.data.DataLoader(DS(enc), batch_size=32)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=2, torch_dtype=torch.bfloat16
).to(device)

model.load_state_dict(torch.load(".cache/best_model_fold1.pt"))
model.eval()

probs = []
with torch.no_grad():
    for batch in loader:
        b = {k:v.to(device) for k,v in batch.items()}
        logits = model(**b).logits
        p = torch.softmax(logits,dim=-1)[:,1]
        probs.append(p.cpu().numpy())

probs = np.concatenate(probs)
pd.DataFrame({"id": test_df.index, "prob": probs}).to_csv("roberta_fold1_pred.csv", index=False)
print("✔ Saved roberta_fold1_pred.csv")
