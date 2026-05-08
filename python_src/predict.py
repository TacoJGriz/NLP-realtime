import onnxruntime as ort
import numpy as np
import json


def load_resources():
    with open("../model_data/vocab.json", "r") as f:
        vocab = json.load(f)
    with open("../model_data/label_map.json", "r") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    return vocab, label_map


def encode_input(text, vocab, seq_len=15):
    tokens = text.lower().split()
    encoded = [vocab.get(t, 1) for t in tokens]
    if len(encoded) < seq_len:
        encoded += [0] * (seq_len - len(encoded))
    return np.array([encoded[:seq_len]], dtype=np.int64)


def run_prediction(model_path):
    vocab, label_map = load_resources()
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    print(f"\n--- Model '{model_path}' is Ready ---")
    print("Type a command (or 'q' to quit):")

    while True:
        user_input = input("> ")
        if user_input.lower() == "q":
            break

        input_tensor = encode_input(user_input, vocab)

        outputs = session.run(None, {input_name: input_tensor})

        logits = np.array(outputs[1] if len(outputs) > 1 else outputs[0])

        pred_idx = np.argmax(logits, axis=1)[0]
        confidence = np.max(np.exp(logits) / np.sum(np.exp(logits)))

        intent = label_map.get(pred_idx, "Unknown")
        print(f"Result: {intent} ({confidence * 100:.2f}%)")


if __name__ == "__main__":
    run_prediction("../model_data/optimizedGRU.onnx")
