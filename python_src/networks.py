from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from datasets import load_dataset, load_from_disk
import json
import sys


def load_snips_hf(
    path="../datasets/local_snips_data", path2="../datasets/local_snips_intent"
):
    print("Loading data...")
    try:
        dataset = load_from_disk(path)
    except FileNotFoundError:
        dataset = load_dataset("AutoIntent/snips", name="default")
        dataset.save_to_disk(path)

    try:
        intents_ds = load_from_disk(path2)
    except FileNotFoundError:
        intents_ds = load_dataset("AutoIntent/snips", name="intents")
        intents_ds.save_to_disk(path2)
    intent_data: Any = intents_ds["intents"]

    sorted_intents = sorted(intent_data, key=lambda x: x["id"])
    intent_names = [item["name"] for item in sorted_intents]

    train_data: Any = dataset["train"]
    texts = train_data["utterance"]
    labels = train_data["label"]

    return texts, labels, intent_names


class SnipsDataset(Dataset):
    def __init__(self, texts, labels, vocab, label_map, seq_len=15):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.label_map = label_map
        self.seq_len = seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = encode_text(self.texts[idx], self.vocab, self.seq_len)

        label = self.labels[idx]

        return torch.tensor(encoded, dtype=torch.long), torch.tensor(
            label, dtype=torch.long
        )


def build_vocab(texts, max_vocab=5000):
    all_words = " ".join(texts).lower().split()
    counts = Counter(all_words)
    vocab = {word: i + 2 for i, (word, _) in enumerate(counts.most_common(max_vocab))}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab


def encode_text(text, vocab, seq_len=15):
    tokens = text.lower().split()
    encoded = [vocab.get(t, 1) for t in tokens]
    if len(encoded) < seq_len:  # This adds padding to the tokens
        encoded += [0] * (seq_len - len(encoded))
    return encoded[:seq_len]


class naiveLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden[-1])


class optimizedGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.gru(embedded)
        return self.fc(hidden[-1])


class earlyExit(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.layer1 = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.layer2 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        self.exit1 = nn.Linear(hidden_dim, output_dim)
        self.exit2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, force_exit=False):
        x = self.embedding(x)
        out, hidden1 = self.layer1(x)
        logits1 = self.exit1(hidden1[-1])

        if force_exit:
            return logits1, logits1, torch.tensor([1], dtype=torch.long)

        _, hidden2 = self.layer2(out)
        logits2 = self.exit2(hidden2[-1])

        return logits1, logits2, torch.tensor([0], dtype=torch.long)


def export_to_onnx(model, model_name, vocab_size, seq_len=15):
    model.eval()
    dummy_input = torch.randint(0, vocab_size, (1, seq_len))

    torch.onnx.export(
        model,
        (dummy_input,),
        f"../model_data/{model_name}.onnx",
        export_params=True,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Model exported to {model_name}.onnx")


def train_model(model, train_loader, epochs=10, lr=1e-3, device=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    if device:
        model.to(device)

    for epoch in range(epochs):
        running_loss = 0.0
        num_batches = 0
        for x_batch, y_batch in train_loader:
            if device:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

            optimizer.zero_grad()
            if isinstance(model, earlyExit):
                logits_early, logits_final, _ = model(x_batch)
                loss = criterion(logits_early, y_batch) + criterion(
                    logits_final, y_batch
                )
            else:
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        epoch_loss = running_loss / max(1, num_batches)
        print(f"Epoch {epoch + 1} complete. Avg loss: {epoch_loss:.4f}")


def create_dataloader(texts, labels, vocab, label_map, batch_size=32):
    dataset = SnipsDataset(texts, labels, vocab, label_map)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":
    texts, labels, names = load_snips_hf()
    vocab = build_vocab(texts)
    label_map = {i: name for i, name in enumerate(names)}

    model_opt = optimizedGRU(len(vocab), 64, 128, len(label_map))
    if len(sys.argv) > 1:
        if sys.argv[1] == "earlyExit":
            model_opt = earlyExit(len(vocab), 64, 128, len(label_map))
        elif sys.argv[1] == "naive":
            model_opt = naiveLSTM(len(vocab), 64, 128, len(label_map))
        elif sys.argv[1] == "optimized":
            model_opt = optimizedGRU(len(vocab), 64, 128, len(label_map))

    train_loader = create_dataloader(texts, labels, vocab, label_map)
    print("Training model: " + model_opt.__class__.__name__)
    train_model(model_opt, train_loader)

    name = ""
    if len(sys.argv) > 2:
        name = "_" + sys.argv[2]

    export_to_onnx(model_opt, model_opt.__class__.__name__ + name, len(vocab))

    with open("../model_data/vocab.json", "w") as f:
        json.dump(vocab, f)
        print("Vocab saved to " + f.name)

    with open("../model_data/label_map.json", "w") as f:
        json.dump(label_map, f)
        print("Label map saved to " + f.name)
