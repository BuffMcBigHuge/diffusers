# gethash.py

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_path", default=None, type=str, required=True, help="Path to the output model.")

    args = parser.parse_args()

    assert args.checkpoint_path is not None, "Must provide a checkpoint path!"

def model_hash(filename):
    try:
        with open(filename, "rb") as file:
            import hashlib
            m = hashlib.sha256()

            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[0:8]
    except FileNotFoundError:
        return 'NOFILE'

print(f"{args.checkpoint_path}");

h = model_hash(args.checkpoint_path);

print(f"Hash: {h}")