import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from cltk.corpus.greek.beta_to_unicode import Replacer
from transformers import TFT5ForConditionalGeneration, T5Tokenizer
import spacy

class TranslationDataset(Dataset):
    def __init__(self, data_dir, source_lang, target_lang, nlp_src, nlp_tgt, max_seq_len):
        self.nlp_source = nlp_src
        self.nlp_target = nlp_tgt
        self.max_seq_len = max_seq_len

        # Read file pairs from the data directory
        self.file_pairs = self.read_file_pairs(data_dir, source_lang, target_lang)

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, index):
        r = Replacer()
        file_pair = self.file_pairs[index]
        with open(file_pair["source"]) as f_src, open(file_pair["target"], "r") as f_tgt:
            source_text = r.beta_code(r"%s" % f_src.read())
            target_text = f_tgt.read()

        # Tokenize the source and target sentences
        source_tokens = self.preprocess(source_text, self.nlp_source)
        target_tokens = self.preprocess(target_text, self.nlp_target)
        # Add special tokens to the input and output sequences
        input_ids = [self.nlp_source.vocab.strings[source_token.text] for source_token in source_tokens]
        labels = [self.nlp_target.vocab.strings[target_token.text] for target_token in target_tokens]

        # Pad or truncate the input and output sequences to max_seq_len
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
        else:
            input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
        if len(labels) > self.max_seq_len:
            labels = labels[:self.max_seq_len]
        else:
            labels = labels + [0] * (self.max_seq_len - len(labels))

        # Create attention masks to ignore padded tokens
        attention_mask = [1] * len(input_ids)

        # Convert to PyTorch tensors and return as dictionary
        return {
            "input_ids": torch.LongTensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.LongTensor(labels),
        }

    def read_file_pairs(self, data_dir, source_lang, target_lang):
        file_pairs = []
        for filename in os.listdir(data_dir):
            if filename.endswith(f"{source_lang}.txt"):
                source_file = filename
                target_file = filename.replace(f"_{source_lang}", f"_{target_lang}")
                file_pairs.append({"source": data_dir + "/" + source_file, "target": data_dir + "/" + target_file})
        return file_pairs

    def preprocess(self, text, nlp):
        doc = nlp(text)
        return [token for token in doc.sents][0]


tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = TFT5ForConditionalGeneration.from_pretrained("t5-small")

src_lang = "gk"
tgt_lang = "eng"
nlp_grc = spacy.load("grc_ud_proiel_md")
nlp_eng = spacy.load("en_core_web_md")

data_dir = "./data/Pairs"
dataset = TranslationDataset(data_dir, src_lang, tgt_lang, nlp_grc, nlp_eng, 512)

train_size = int(0.7 * len(dataset))
eval_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - eval_size
translation_test_dataset, eval_dataset = random_split(dataset, [train_size + test_size, eval_size])
translation_dataset, test_dataset = random_split(dataset, [train_size, test_size])


translation_dataloader = DataLoader(translation_dataset, batch_size=8, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=8, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Set up the optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.095)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for epoch in range(20):
    running_loss = 0.0
    for batch in translation_dataloader:
        model.train()
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        model.eval()
        with torch.no_grad():
            eval_loss = 0.0
            eval_correct = 0
            eval_total = 0
            for batch in eval_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                eval_loss += loss.item()
                eval_correct += (logits.argmax(-1) == labels).sum().item()
                eval_total += labels.numel()

            eval_accuracy = eval_correct / eval_total
            print(f'Epoch {epoch + 1}, Eval Loss: {eval_loss:.3f}, Eval Accuracy: {eval_accuracy:.3f}')

        model.train()

    # Print the average loss for the epoch
    print(f"Epoch {epoch+1} loss: {running_loss/len(translation_dataloader)}")

model.save_pretrained("./models/t5-gk-en")