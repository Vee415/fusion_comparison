# Late-2D Fusion Deploy (C++ TensorRT)

Deploys the late_2d fusion model on a Jetson using TensorRT FP16 engines.
Two backbones (cam + lid) run independently, then merge/NMS on CPU.

## Build

```bash
# On the Jetson:
cd ~/fusion_deploy
cmake . && make
```

## Convert ONNX to TRT engines (already done)

```bash
export PATH=/usr/src/tensorrt/bin:$PATH
trtexec --onnx=onnx/late_2d_cam.onnx --saveEngine=build/late_2d_cam.engine --fp16
trtexec --onnx=onnx/late_2d_lid.onnx --saveEngine=build/late_2d_lid.engine --fp16
```

## Run

```bash
./late_2d_deploy \
  --cam_engine build/late_2d_cam.engine \
  --lid_engine build/late_2d_lid.engine \
  --image data/image_2/000001.png \
  --velodyne data/velodyne/000001.bin \
  --calib data/calib/000001.txt \
  --output result.png
```

Output: `result.png` with bounding boxes (blue=cam, green=lidar, red=merged).

## Architecture

```
camera image (384x1280 RGB)          LiDAR points (N,4)
        |                                    |
   cam_preprocess (CUDA)              lidar_render (CUDA)
   BGR->RGB->resize->/255->CHW        project to image -> [depth, height, intensity]
        |                                    |
  cam TRT engine (FP16)              lid TRT engine (FP16)
   ~4.4ms, 25MB engine               ~4.4ms, 25MB engine
        |                                    |
    heat/off/size (24x80)           heat/off/size (24x80)
        |                                    |
    decode (CPU)                     decode (CPU)
    sigmoid->topk40->boxes           sigmoid->topk40->boxes
        |                                    |
    NMS 0.45 (CPU)                   NMS 0.45 (CPU)
        |                                    |
        +------------- merge (CPU) ----------+
                      | greedy IoU 0.5, mean score
                    NMS 0.5
                      |
                merged detections
                      |
              draw boxes -> result.png
```

## Performance (Jetson Orin, FP16)

| Metric | Value |
|---|---|
| cam engine inference | ~4.4 ms |
| lid engine inference | ~4.4 ms |
| total inference | ~8.8 ms |
| preprocess (CUDA) | ~1 ms |
| decode+NMS+merge (CPU) | ~1 ms |
| **total pipeline** | **~11 ms (90 FPS)** |

## Files

- `main.cpp` — CLI entry, orchestrates the full pipeline
- `trt_engine.h/.cpp` — TRT engine load, buffer alloc, enqueueV3
- `preprocess.cu` — CUDA kernels: cam (resize+normalize) + lidar (project+render)
- `decode.h/.cpp` — sigmoid → topk → box reconstruction
- `nms.h/.cpp` — greedy IoU NMS
- `merge.h/.cpp` — late fusion: greedy match, mean score, cam box retained
- `calib_parser.h/.cpp` — KITTI calib txt parser (P2, R0, Tr_velo_to_cam)