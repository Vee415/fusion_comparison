// TensorRT engine wrapper implementation (TRT 10.x API).
#include "trt_engine.h"
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>
#include <cstring>

void TrtLogger::log(ILogger::Severity severity, const char* msg) noexcept {
    if (severity <= ILogger::Severity::kWARNING)
        std::cerr << "[TRT] " << msg << std::endl;
}

static TrtLogger g_logger;

TrtEngine::TrtEngine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "ERROR: cannot open engine: " << engine_path << std::endl;
        return;
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> data(size);
    file.read(data.data(), size);

    runtime_ = nvinfer1::createInferRuntime(g_logger);
    engine_ = runtime_->deserializeCudaEngine(data.data(), size);
    if (!engine_) {
        std::cerr << "ERROR: failed to deserialize engine" << std::endl;
        return;
    }
    context_ = engine_->createExecutionContext();
    n_bindings_ = engine_->getNbIOTensors();

    // Build a map: tensor name -> index in gpu_buffers_
    for (int i = 0; i < n_bindings_; i++) {
        const char* name = engine_->getIOTensorName(i);
        nvinfer1::Dims dims = engine_->getTensorShape(name);
        nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(name);

        Binding b;
        b.name = name;
        b.count = 1;
        for (int d = 0; d < dims.nbDims; d++) {
            b.dims.push_back(dims.d[d]);
            b.count *= dims.d[d];
        }
        b.bytes = b.count * sizeof(float);
        b.is_input = (mode == nvinfer1::TensorIOMode::kINPUT);

        void* gpu_buf = nullptr;
        cudaMalloc(&gpu_buf, b.bytes);
        gpu_buffers_.push_back(gpu_buf);
        name_to_idx_[name] = i;

        if (b.is_input) inputs_.push_back(b);
        else outputs_.push_back(b);
    }
    std::cout << "Loaded engine: " << engine_path << " (" << inputs_.size()
              << " inputs, " << outputs_.size() << " outputs)" << std::endl;
}

TrtEngine::~TrtEngine() {
    for (auto& buf : gpu_buffers_) if (buf) cudaFree(buf);
    if (context_) delete context_;
    if (engine_) delete engine_;
    if (runtime_) delete runtime_;
}

void TrtEngine::infer(const std::vector<float*>& inputs, std::vector<float*>& outputs) {
    // H2D + set tensor addresses by name (TRT 10.x)
    for (size_t i = 0; i < inputs_.size(); i++) {
        int idx = name_to_idx_[inputs_[i].name];
        cudaMemcpy(gpu_buffers_[idx], inputs[i], inputs_[i].bytes, cudaMemcpyHostToDevice);
        context_->setTensorAddress(inputs_[i].name.c_str(), gpu_buffers_[idx]);
    }
    for (size_t i = 0; i < outputs_.size(); i++) {
        int idx = name_to_idx_[outputs_[i].name];
        context_->setTensorAddress(outputs_[i].name.c_str(), gpu_buffers_[idx]);
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    context_->enqueueV3(stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    // D2H
    for (size_t i = 0; i < outputs_.size(); i++) {
        int idx = name_to_idx_[outputs_[i].name];
        cudaMemcpy(outputs[i], gpu_buffers_[idx], outputs_[i].bytes, cudaMemcpyDeviceToHost);
    }
}