import time
from typing import Any
import onnxruntime as ort
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import json
from torch.utils.data import DataLoader
from datasets import load_from_disk
from networks import SnipsDataset
import sys
import csv


def run_python_baseline(
    model_path, test_file="../test_phrases.txt", vocab_path="../model_data/vocab.json"
):
    with open(vocab_path, "r") as f:
        vocab = json.load(f)

    with open(test_file, "r") as f:
        phrases = [line.strip() for line in f.readlines()][:1400]

    session = ort.InferenceSession(model_path)
    results = []

    def tokenize(text):
        tokens = text.lower().split()
        encoded = [vocab.get(t, 1) for t in tokens]
        return (encoded + [0] * 15)[:15]

    print("Starting Python Baseline...")
    for i, phrase in enumerate(phrases):
        start = time.perf_counter_ns()
        vec = np.array([tokenize(phrase)], dtype=np.int64)

        session.run(None, {"input": vec})
        end = time.perf_counter_ns()

        results.append([i, (end - start) / 1e6])  # ms

    with open(
        f"../results/{model_path[model_path.rindex('/') : model_path.rindex('.')]}_python.csv",
        "w",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "latency_ms"])
        writer.writerows(results)
        print("Python results saved to " + f.name)


def prepare_test_data(path="../datasets/local_snips_data", vocab=None, label_map=None):
    dataset = load_from_disk(path)
    test_data: Any = dataset["test"]

    test_dataset = SnipsDataset(
        test_data["utterance"], test_data["label"], vocab, label_map=label_map
    )

    return DataLoader(test_dataset, batch_size=64, shuffle=False)


def evaluate_accuracy(session, test_loader):
    all_preds = []
    all_labels = []

    input_name = session.get_inputs()[0].name

    for x_batch, y_batch in test_loader:
        onnx_inputs = {input_name: x_batch.numpy()}
        logits = session.run(None, onnx_inputs)[0]

        preds = np.argmax(logits, axis=1)
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

    print(classification_report(all_labels, all_preds))
    return confusion_matrix(all_labels, all_preds)


def run_accuracy_test(onnx_path, test_loader):
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    all_preds = []
    all_labels = []

    print(f"Starting accuracy test for: {onnx_path}...")

    for x_batch, y_batch in test_loader:
        inputs = {input_name: x_batch.numpy().astype(np.int64)}
        outputs = session.run(None, inputs)
        logits = np.array(outputs[0])

        preds = np.argmax(logits, axis=1)

        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds))

    return confusion_matrix(all_labels, all_preds)


def test_inference(onnx_path):
    print("Testing " + onnx_path)
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    test_data = np.random.randint(0, 5000, (1, 15)).astype(np.int64)

    start = time.perf_counter()
    for _ in range(100):
        session.run(None, {input_name: test_data})
    end = time.perf_counter()

    print(f"Average Inference Time for {onnx_path}: {(end - start) / 100:.6f}s")


if __name__ == "__main__":
    for path in ["optimizedGRU.onnx", "earlyExit.onnx", "naiveLSTM.onnx"]:
        if len(sys.argv) > 1:
            run_python_baseline("../model_data/" + path)

        else:
            test_inference("../model_data/" + path)

            with open("../model_data/vocab.json", "r") as f:
                vocab = json.load(f)
                print("Loaded vocal from " + f.name)

            with open("../model_data/label_map.json", "r") as f:
                loaded_map = json.load(f)
                label_map = {int(k): v for k, v in loaded_map.items()}
                print("Loaded label map from " + f.name)

            test_loader = prepare_test_data(
                "../datasets/local_snips_data", vocab, label_map
            )
            matrix = run_accuracy_test("../model_data/" + path, test_loader)
