import os
import shutil

# Root directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

def main():
    print("=" * 70)
    print("Weights Folder Re-organization")
    print("=" * 70)

    weights_dir = os.path.join(_PROJECT_ROOT, "weights")
    legacy_dir = os.path.join(weights_dir, "legacy_dual_model")
    os.makedirs(legacy_dir, exist_ok=True)

    # Files to move to legacy
    legacy_files = [
        "sakku_int8.xml",
        "sakku_int8.bin",
        "sakku_embedding.xml",
        "sakku_embedding.bin"
    ]

    print("Moving legacy dual-model files to weights/legacy_dual_model/ ...")
    moved_count = 0
    for filename in legacy_files:
        src = os.path.join(weights_dir, filename)
        dst = os.path.join(legacy_dir, filename)
        if os.path.exists(src):
            try:
                shutil.move(src, dst)
                print(f"  Moved: {filename} -> weights/legacy_dual_model/{filename}")
                moved_count += 1
            except Exception as e:
                print(f"  Error moving {filename}: {e}")
        else:
            print(f"  Skipped: {filename} (not found in weights/ root)")

    print(f"\nSuccessfully moved {moved_count} legacy files.")
    print("Active root weights folder now only contains your optimized multi-output models.")
    print("=" * 70)

if __name__ == "__main__":
    main()
