import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plots(cpp_df, py_df, name):
    py_df["Language"] = "Python"
    cpp_df["Language"] = "C++ (RT-Optimized)"
    df = pd.concat([cpp_df, py_df])

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    sns.lineplot(data=df, x="iteration", y="latency_ms", hue="Language", alpha=0.7)
    plt.title("Inference Jitter: C++ vs. Python for " + name + " model")
    plt.ylabel("Latency (ms)")
    plt.xlabel("Iteration")

    plt.subplot(2, 1, 2)
    sns.violinplot(data=df, x="Language", y="latency_ms", inner="quart")
    plt.title("Latency Distribution & Tail Latency")
    plt.ylabel("Latency (ms)")

    plt.tight_layout()
    plt.savefig("../results/charts/" + name + "benchmark_comparison.png", dpi=300)
    plt.show()

    stats = df.groupby("Language")["latency_ms"].agg(["mean", "std", "max", "min"])
    stats["jitter"] = stats["max"] - stats["min"]
    print(stats)


cpp_gru = pd.read_csv("../results/optimizedGRU_cpp.csv")
py_gru = pd.read_csv("../results/optimizedGRU_python.csv")
cpp_ee = pd.read_csv("../results/earlyExit_cpp.csv")
py_ee = pd.read_csv("../results/earlyExit_python.csv")
cpp_lstm = pd.read_csv("../results/naiveLSTM_cpp.csv")
py_lstm = pd.read_csv("../results/naiveLSTM_python.csv")

for cpp, py, name in [
    (cpp_gru, py_gru, "GRU"),
    (cpp_ee, py_ee, "Early Exit"),
    (cpp_lstm, py_lstm, "LSTM"),
]:
    print("Making chart for " + name)
    plots(cpp, py, name)
    input("Next chart...")
