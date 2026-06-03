import os
import sys
import time
import numpy as np

def benchmark_openvino_model(xml_path, dummy_input, num_runs=50, is_multi=False):
    if not os.path.exists(xml_path):
        return None
    import openvino as ov
    core = ov.Core()
    model = core.read_model(xml_path)
    # Reshape to static shape to ensure fair and fast execution
    input_name = model.inputs[0].any_name
    model.reshape({input_name: [1, 3, 640, 640]})
    compiled = core.compile_model(model, "CPU")
    request = compiled.create_infer_request()
    
    # Warmup
    for _ in range(5):
        request.infer({0: dummy_input})
        
    times = []
    for _ in range(num_runs):
        t0 = time.time()
        request.infer({0: dummy_input})
        times.append(time.time() - t0)
        
    return np.mean(times) * 1000  # ms

def main():
    print("=" * 80)
    print("FPS & Computational Latency Benchmark: Dual-Model vs. Multi-Output Configurations")
    print("=" * 80)

    # File paths
    ov_det_path = os.path.normpath("weights/sakku_int8.xml")
    ov_emb_path = os.path.normpath("weights/sakku_embedding.xml")
    onnx_multi_path = os.path.normpath("weights/sakku_multi_output.onnx")
    ov_multi_fp32_path = os.path.normpath("weights/sakku_multi_output.xml")
    ov_multi_int8_path = os.path.normpath("weights/sakku_multi_output_int8.xml")

    # Check dependencies
    has_ov = True
    try:
        import openvino as ov
        print(f"OpenVINO version: {ov.__version__}")
    except ImportError:
        has_ov = False
        print("WARNING: openvino is not installed.")

    has_ort = True
    try:
        import onnxruntime as ort
        print(f"ONNX Runtime version: {ort.__version__}")
    except ImportError:
        has_ort = False
        print("WARNING: onnxruntime is not installed.")

    if not has_ov or not has_ort:
        print("ERROR: Both openvino and onnxruntime must be installed to run the benchmark.")
        return

    dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
    num_runs = 50
    results = {}

    # ----------------------------------------------------
    # Benchmark 1: OpenVINO Dual-Model (INT8 Detection + FP32 Embedding)
    # ----------------------------------------------------
    print("\n[1/4] Benchmarking OpenVINO Dual-Model pipeline...")
    if os.path.exists(ov_det_path) and os.path.exists(ov_emb_path):
        try:
            det_ms = benchmark_openvino_model(ov_det_path, dummy_input, num_runs)
            emb_ms = benchmark_openvino_model(ov_emb_path, dummy_input, num_runs)
            total_ms = det_ms + emb_ms
            results["1. OpenVINO Dual-Model (INT8 + FP32)"] = {
                "det_ms": det_ms,
                "emb_ms": emb_ms,
                "total_ms": total_ms,
                "fps": 1000.0 / total_ms
            }
            print(f"      Detection: {det_ms:.2f} ms | Embedding: {emb_ms:.2f} ms | Total: {total_ms:.2f} ms ({1000.0/total_ms:.1f} FPS)")
        except Exception as e:
            print(f"      Failed: {e}")
    else:
        print("      Skipped: Model files not found.")

    # ----------------------------------------------------
    # Benchmark 2: ONNX Runtime Single-Model Multi-Output (FP32)
    # ----------------------------------------------------
    print("\n[2/4] Benchmarking ONNX Runtime Multi-Output model (FP32)...")
    if os.path.exists(onnx_multi_path):
        try:
            ort_sess = ort.InferenceSession(onnx_multi_path, providers=['CPUExecutionProvider'])
            input_name = ort_sess.get_inputs()[0].name
            # Warmup
            for _ in range(5):
                ort_sess.run(None, {input_name: dummy_input})
            
            times = []
            for _ in range(num_runs):
                t0 = time.time()
                ort_sess.run(None, {input_name: dummy_input})
                times.append(time.time() - t0)
            
            total_ms = np.mean(times) * 1000
            results["2. ONNX Runtime Multi-Output (FP32)"] = {
                "total_ms": total_ms,
                "fps": 1000.0 / total_ms
            }
            print(f"      Total Latency: {total_ms:.2f} ms ({1000.0/total_ms:.1f} FPS)")
        except Exception as e:
            print(f"      Failed: {e}")
    else:
        print("      Skipped: Model file weights/sakku_multi_output.onnx not found.")

    # ----------------------------------------------------
    # Benchmark 3: OpenVINO Single-Model Multi-Output (FP32)
    # ----------------------------------------------------
    print("\n[3/4] Benchmarking OpenVINO Multi-Output model (FP32)...")
    if os.path.exists(ov_multi_fp32_path):
        try:
            total_ms = benchmark_openvino_model(ov_multi_fp32_path, dummy_input, num_runs, is_multi=True)
            results["3. OpenVINO Multi-Output (FP32)"] = {
                "total_ms": total_ms,
                "fps": 1000.0 / total_ms
            }
            print(f"      Total Latency: {total_ms:.2f} ms ({1000.0/total_ms:.1f} FPS)")
        except Exception as e:
            print(f"      Failed: {e}")
    else:
        print("      Skipped: Model file weights/sakku_multi_output.xml not found.")

    # ----------------------------------------------------
    # Benchmark 4: OpenVINO Single-Model Multi-Output (INT8)
    # ----------------------------------------------------
    print("\n[4/4] Benchmarking OpenVINO Multi-Output model (INT8)...")
    if os.path.exists(ov_multi_int8_path):
        try:
            total_ms = benchmark_openvino_model(ov_multi_int8_path, dummy_input, num_runs, is_multi=True)
            results["4. OpenVINO Multi-Output (INT8)"] = {
                "total_ms": total_ms,
                "fps": 1000.0 / total_ms
            }
            print(f"      Total Latency: {total_ms:.2f} ms ({1000.0/total_ms:.1f} FPS)")
        except Exception as e:
            print(f"      Failed: {e}")
    else:
        print("      Skipped: Model file weights/sakku_multi_output_int8.xml not found.")
        print("      Run quantize_multi_output.py first to quantize to INT8.")

    # ----------------------------------------------------
    # Print Comparison Table
    # ----------------------------------------------------
    print("\n" + "=" * 80)
    print(" FINAL LATENCY & FPS BENCHMARK COMPARISON")
    print("=" * 80)
    print(f"  {'Pipeline/Model Configuration':<48} | {'Latency (ms)':<12} | {'Speed (FPS)':<10}")
    print("-" * 80)
    
    for name, metrics in results.items():
        print(f"  {name:<48} | {metrics['total_ms']:>10.2f} ms | {metrics['fps']:>8.1f} FPS")
    print("=" * 80)
    
    # Highlight recommendation
    if "4. OpenVINO Multi-Output (INT8)" in results and "1. OpenVINO Dual-Model (INT8 + FP32)" in results:
        t_dual = results["1. OpenVINO Dual-Model (INT8 + FP32)"]["total_ms"]
        t_multi = results["4. OpenVINO Multi-Output (INT8)"]["total_ms"]
        ov_speedup = t_dual / t_multi
        print(f"RECOMMENDATION:")
        print(f"- Running multi-output OpenVINO in INT8 gives you a {ov_speedup:.2f}x speedup compared to")
        print(f"  the dual-model pipeline because the backbone features are shared.")
    else:
        print(f"RECOMMENDATION:")
        print(f"- Running multi-output OpenVINO in INT8 is highly recommended for best performance on CPU/iGPU.")
    print("=" * 80)

if __name__ == "__main__":
    main()
