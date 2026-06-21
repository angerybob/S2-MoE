import struct, sys

def read_u32(f): return struct.unpack("<I", f.read(4))[0]
def read_u64(f): return struct.unpack("<Q", f.read(8))[0]
def read_str(f):
    n = read_u64(f)
    return f.read(n).decode("utf-8", errors="replace")

# GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md (layout widely used)
GGUF_MAGIC = b"GGUF"
GGUF_VERSION_MIN = 1

# value types (only need to skip)
GGUF_TYPE_U8  = 0
GGUF_TYPE_I8  = 1
GGUF_TYPE_U16 = 2
GGUF_TYPE_I16 = 3
GGUF_TYPE_U32 = 4
GGUF_TYPE_I32 = 5
GGUF_TYPE_F32 = 6
GGUF_TYPE_BOOL= 7
GGUF_TYPE_STR = 8
GGUF_TYPE_ARR = 9
GGUF_TYPE_U64 = 10
GGUF_TYPE_I64 = 11
GGUF_TYPE_F64 = 12

def skip_value(f, vtype):
    if vtype in (GGUF_TYPE_U8, GGUF_TYPE_I8, GGUF_TYPE_BOOL):
        f.read(1)
    elif vtype in (GGUF_TYPE_U16, GGUF_TYPE_I16):
        f.read(2)
    elif vtype in (GGUF_TYPE_U32, GGUF_TYPE_I32, GGUF_TYPE_F32):
        f.read(4)
    elif vtype in (GGUF_TYPE_U64, GGUF_TYPE_I64, GGUF_TYPE_F64):
        f.read(8)
    elif vtype == GGUF_TYPE_STR:
        _ = read_str(f)
    elif vtype == GGUF_TYPE_ARR:
        elem_type = read_u32(f)
        n = read_u64(f)
        for _ in range(n):
            skip_value(f, elem_type)
    else:
        raise ValueError(f"Unknown GGUF value type {vtype}")

def main(path, needle):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != GGUF_MAGIC:
            raise SystemExit("Not a GGUF file")
        version = read_u32(f)
        if version < GGUF_VERSION_MIN:
            raise SystemExit(f"Unsupported GGUF version {version}")
        n_tensors = read_u64(f)
        n_kv = read_u64(f)

        # skip kv
        for _ in range(n_kv):
            _key = read_str(f)
            vtype = read_u32(f)
            skip_value(f, vtype)

        found = False
        for _ in range(n_tensors):
            name = read_str(f)
            n_dims = read_u32(f)
            dims = [read_u64(f) for _ in range(n_dims)]
            ggml_type = read_u32(f)
            _offset = read_u64(f)  # data offset
            if needle == "" or needle == name:
                print(f"{name}\tndims={n_dims}\tshape={dims}\tggml_type={ggml_type}")
                found = True
                if needle:
                    break

        if needle and not found:
            print("NOT FOUND:", needle, file=sys.stderr)
            sys.exit(2)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python gguf_shape.py <model.gguf> [tensor_name]", file=sys.stderr)
        sys.exit(1)
    path = sys.argv[1]
    needle = sys.argv[2] if len(sys.argv) >= 3 else ""
    main(path, needle)
