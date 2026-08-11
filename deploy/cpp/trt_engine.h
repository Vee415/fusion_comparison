// TensorRT engine wrapper: load .engine, allocate buffers, run inference.
#pragma once
#include <NvInfer.h>
#include <string>
#include <vector>
#include <map>

class TrtEngine {
public:
    // I/O metadata
    struct Binding {
        std::string name;
        std::vector<int> dims;   // e.g. [1,3,384,1280] or [1,1,24,80]
        size_t count;            // total elements
        size_t bytes;            // total bytes (count * sizeof(float))
        bool is_input;
    };

    TrtEngine(const std::string& engine_path);
    ~TrtEngine();

    // Run inference. inputs: maps binding name -> host float* data.
    // Outputs are copied to the provided output host buffers (keyed by name).
    void infer(const std::vector<float*>& inputs, std::vector<float*>& outputs);

    const std::vector<Binding>& get_inputs() const { return inputs_; }
    const std::vector<Binding>& get_outputs() const { return outputs_; }

private:
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    std::vector<void*> gpu_buffers_;
    std::vector<Binding> inputs_;
    std::vector<Binding> outputs_;
    std::map<std::string, int> name_to_idx_;
    int n_bindings_ = 0;
};

// Logger for TensorRT
class TrtLogger : public nvinfer1::ILogger {
    void log(ILogger::Severity severity, const char* msg) noexcept override;
};