from datasets import load_from_disk


def dump_snips_test_data(
    dataset_path="../datasets/local_snips_data/", output_file="../test_phrases.txt"
):
    """Dumps the test split of the SNIPS dataset to a text file."""
    dataset = load_from_disk(dataset_path)
    test_data = dataset["test"]["utterance"]

    with open(output_file, "w") as f:
        for utterance in test_data:
            f.write(f"{utterance}\n")
    print(f"Successfully dumped {len(test_data)} phrases to {output_file}")


if __name__ == "__main__":
    dump_snips_test_data()
