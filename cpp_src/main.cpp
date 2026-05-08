#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>
#include <time.h>
#include <pthread.h>
#include <sys/mman.h>
#include <iostream>
#include <sched.h>
#include <unistd.h>
#include <cstdio>

using json = nlohmann::json;
using namespace std;

#define STACK (1024 * 1024)

// We want the program to run on a single core that we have already determined
// The core is isolated so only this program will run on it
void stick_to_core(int core_id) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);

  pthread_t current_thread = pthread_self();

  int result = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
 
  if (result != 0) {
    cerr << "Error setting thread affinity!" << endl;
  } else {
    cout << "Thread pinned to Core " << core_id << endl;
  }
}

// Tokenize input to a set length for determinism
vector<int64_t> tokenize(string text, const unordered_map<string, int>& vocab, size_t seq_len = 15) {
  vector<int64_t> encoded;
  stringstream ss(text);
  string token;

  while (ss >> token) {
    transform(token.begin(), token.end(), token.begin(), ::tolower);
    if (vocab.count(token)) {
      encoded.push_back(vocab.at(token));
    } else {
      encoded.push_back(1); // <UNK> token
    }
  }

  // Padding/Truncating
  while (encoded.size() < seq_len) {
    encoded.push_back(0); // <PAD> token
  }
  if (encoded.size() > seq_len) {
    encoded.resize(seq_len);
  }

  return encoded;
}


int main(int argc, char* argv[]) {
  string phrase = "Will it rain today?";
  string path = "../model_data/";
  string model = "optimizedGRU.onnx";
  int core_id = 3;
  int priority = 80;
  bool testing = false;

  int opt;
  while ((opt = getopt(argc, argv, "t:m:l:c:p:d:")) != -1) {
    switch (opt) {
      case 't': phrase = optarg; break;
      case 'm': model = optarg; break;
      case 'l': path = optarg; break;
      case 'c': core_id = stoi(optarg); break;
      case 'p': priority = stoi(optarg); break;
      case 'd': testing = true; break; 
      default:
        cerr << "Usage: " << argv[0] << " [-t phrase] [-m model] [-l path] [-c core] [-p priority]" << endl;
        return 1;
    }
  }

  // Our program will live in memory while it runs
  if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
    perror("mlockall failed. Try with sudo");
  } else {
    cout << "Memory locked into RAM successfully." << endl;
  }

  // Prefault the stack so the memory is "warmed up"
  unsigned char dummy[STACK];
  std::memset(dummy, 0, STACK);

  // Changes scheduler to FIFO
  struct sched_param param;
  param.sched_priority = priority; // 1-99

  if (sched_setscheduler(0, SCHED_FIFO, &param) != 0) {
      perror("sched_setscheduler failed. Try with sudo");
  } else {
      cout << "Scheduler set to SCHED_FIFO with priority " << param.sched_priority << endl;
  }

  stick_to_core(core_id);

  struct timespec start, end;
  ifstream vocab_file(path + "vocab.json");
  if (!vocab_file.is_open()) {
    cerr << "Could not open vocab.json!" << endl;
    return 1;
  }
  unordered_map<string, int> vocab;
  json j;
  vocab_file >> j;
  vocab = j.get<unordered_map<string, int>>();

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "RT_Classifier");
  Ort::SessionOptions session_options;
  // do NOT parallelize the ONNX model. We want to run everything on a ssingle core for a more accurate simulation and less jitter
  session_options.SetIntraOpNumThreads(1);
  Ort::Session session(env, (path + model).c_str(), session_options);
  if (testing) {
    ifstream file("../test_phrases.txt");
    string s;
    
    cout << "Running testing loop..." << endl;

    string name = "../results/" + model + "_cpp.csv";
    remove(name.c_str());
    fstream fout;
    fout.open(name, ios::out | ios::app);
    fout << "iteration,latency_ms" << endl;
    
    for (int i = 0; getline(file, s); i++) {
      clock_gettime(CLOCK_MONOTONIC, &start);
      string input_text = s;

      vector<int64_t> input_tensor_values = tokenize(input_text, vocab);

      for (auto i : input_tensor_values) cout << i << " ";
      cout << endl;

      auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      vector<int64_t> input_shape = {1, 15};

      Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(), 
        input_shape.data(), input_shape.size()
      );

      const char* input_names[] = {"input"};
      const char* output_names[] = {"output"}; // Use "logits_final" for earlyExit

      auto output_tensors = session.Run(
        Ort::RunOptions{nullptr}, 
        input_names, &input_tensor, 1, 
        output_names, 1
      );

      float* float_array = output_tensors.front().GetTensorMutableData<float>();
      auto results_count = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();

      int pred_idx = distance(float_array, max_element(float_array, float_array + results_count));

      clock_gettime(CLOCK_MONOTONIC, &end);
      double time_taken = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
      fout << i << "," << time_taken / 1e6 << endl;
    }
  } else {
    clock_gettime(CLOCK_MONOTONIC, &start);
    string input_text = phrase;
    vector<int64_t> input_tensor_values = tokenize(input_text, vocab);

    cout << "Encoded '" << input_text << "': ";
    for (auto i : input_tensor_values) cout << i << " ";
    cout << endl;

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    vector<int64_t> input_shape = {1, 15};

    Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
      memory_info, input_tensor_values.data(), input_tensor_values.size(), 
      input_shape.data(), input_shape.size()
    );

    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"}; // Use "logits_final" for earlyExit

    auto output_tensors = session.Run(
      Ort::RunOptions{nullptr}, 
      input_names, &input_tensor, 1, 
      output_names, 1
    );

    float* float_array = output_tensors.front().GetTensorMutableData<float>();
    auto results_count = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();

    int pred_idx = distance(float_array, max_element(float_array, float_array + results_count));

    cout << "Predicted Intent Index: " << pred_idx << endl;

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec) * 1e9 + (end.tv_nsec - start.tv_nsec);
    cout << "Time passed from input: " << time_taken / 1e6 << " ms" << endl;
  }
}
