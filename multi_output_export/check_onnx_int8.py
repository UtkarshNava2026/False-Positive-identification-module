import os
import sys
import time
import numpy as np

def main():
    print("=" * 60)
    print("ONNX Runtime INT8 Compatibility & Latency Benchmark")
    print("=" * 60)

    try:
        import onnxruntime as ort
        print(f"ONNX Runtime version: {ort.__version__}")
        print(f"Available Execution Providers: {ort.get_available_providers()}")
    except ImportError:
        print("ERROR: onnxruntime is not installed in the current environment.")
        print("Please install it with: pip install onnxruntime")
        return

    # Check for models
    fp32_path = os.path.normpath("weights/backups/sakku_fp32.onnx")
    int8_path = os.path.normpath("development_tools/int8 1.onnx")

    print(f"Checking for FP32 model: {fp32_path} -> {'Found' if os.path.exists(fp32_path) else 'Missing'}")
    print(f"Checking for INT8 model: {int8_path} -> {'Found' if os.path.exists(int8_path) else 'Missing'}")

    if not os.path.exists(fp32_path):
        print("ERROR: FP32 model is missing. Cannot benchmark.")
        return
    if not os.path.exists(int8_path):
        print("ERROR: INT8 model is missing. Cannot benchmark.")
        return

    # Load sessions
    print("\nLoading models into ONNX Runtime...")
    try:
        t0 = time.time()
        fp32_sess = ort.InferenceSession(fp32_path, providers=['CPUExecutionProvider'])
        print(f"FP32 model loaded in {time.time() - t0:.2f}s")
    except Exception as e:
        print(f"Error loading FP32 model: {e}")
        return

    try:
        t0 = time.time()
        int8_sess = ort.InferenceSession(int8_path, providers=['CPUExecutionProvider'])
        print(f"INT8 model loaded in {time.time() - t0:.2f}s")
    except Exception as e:
        print(f"Error loading INT8 model: {e}")
        return

    # Setup dummy input
    input_name_fp32 = fp32_sess.get_inputs()[0].name
    input_name_int8 = int8_sess.get_inputs()[0].name
    dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)

    # Warmup
    print("\nWarming up models (10 iterations each)...")
    for _ in range(10):
        fp32_sess.run(None, {input_name_fp32: dummy_input})
        int8_sess.run(None, {input_name_int8: dummy_input})

    # Benchmark FP32
    num_runs = 50
    print(f"Benchmarking FP32 model ({num_runs} iterations)...")
    fp32_times = []
    for _ in range(num_runs):
        t_start = time.time()
        fp32_sess.run(None, {input_name_fp32: dummy_input})
        fp32_times.append(time.time() - t_start)
    fp32_avg = np.mean(fp32_times) * 1000  # ms
    fp32_fps = 1.0 / np.mean(fp32_times)

    # Benchmark INT8
    print(f"Benchmarking INT8 model ({num_runs} iterations)...")
    int8_times = []
    for _ in range(num_runs):
        t_start = time.time()
        int8_sess.run(None, {input_name_int8: dummy_input})
        int8_times.append(time.time() - t_start)
    int8_avg = np.mean(int8_times) * 1000  # ms
    int8_fps = 1.0 / np.mean(int8_times)

    # Report results
    print("\n" + "=" * 50)
    print(" BENCHMARK RESULTS")
    print("=" * 50)
    print(f"FP32 Avg Latency: {fp32_avg:.2f} ms ({fp32_fps:.1f} FPS)")
    print(f"INT8 Avg Latency: {int8_avg:.2f} ms ({int8_fps:.1f} FPS)")
    
    speedup = fp32_avg / int8_avg
    print(f"Speedup factor:   {speedup:.2f}x")
    print("-" * 50)

    if speedup > 1.05:
        print("SUCCESS: Your PC supports true INT8 acceleration in ONNX Runtime!")
        print(f"You get a {speedup:.2f}x speedup running in INT8.")
    else:
        print("NOTE: INT8 is compatible and runs successfully, but does not provide a speedup over FP32 on this CPU provider.")
        print("This is normal for ONNX Runtime on CPUs without AVX-512 VNNI or AMX instructions.")
        print("To get hardware-accelerated INT8 speedup on this machine, it is highly recommended to use the OpenVINO backend instead.")
    print("=" * 50)

if __name__ == "__main__":
    main()
