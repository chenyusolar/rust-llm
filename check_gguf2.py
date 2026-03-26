import struct

with open(r'e:\models\qwen\Qwen3.5-9B-Q4_K_M.gguf', 'rb') as f:
    # Read header
    magic = f.read(4)
    print(f"Magic: {magic}")

    version = struct.unpack('<I', f.read(4))[0]
    print(f"Version: {version}")

    tensor_count = struct.unpack('<Q', f.read(8))[0]
    print(f"Tensor count: {tensor_count}")

    metadata_count = struct.unpack('<Q', f.read(8))[0]
    print(f"Metadata count: {metadata_count}")

    print(f"\n=== After header (pos 24) ===")

    # Try to read metadata entries
    print("\n=== Reading metadata entries ===")
    for i in range(5):
        pos = f.tell()
        print(f"\nMetadata entry {i} at pos {pos}:")

        # Read key as uint64 string (llama.cpp format)
        key_len_bytes = f.read(8)
        key_len = struct.unpack('<Q', key_len_bytes)[0]
        print(f"  key_len (uint64): {key_len}")

        if key_len > 1000:
            # Try uint32 instead
            f.seek(pos)
            key_len_bytes = f.read(4)
            key_len = struct.unpack('<I', key_len_bytes)[0]
            print(f"  key_len (uint32): {key_len}")
            f.seek(pos + 4)

        key_data = f.read(key_len)
        print(f"  key: {key_data}")

        # Read value_type
        value_type_bytes = f.read(4)
        value_type = struct.unpack('<I', value_type_bytes)[0]
        print(f"  value_type: {value_type}")

        # Read value based on type
        if value_type == 0:  # UINT8
            print(f"  Skipping 1 byte for UINT8")
        elif value_type == 1:  # INT8
            print(f"  Skipping 1 byte for INT8")
        elif value_type == 2:  # UINT16
            print(f"  Skipping 2 bytes for UINT16")
        elif value_type == 3:  # INT16
            print(f"  Skipping 2 bytes for INT16")
        elif value_type == 4:  # UINT32
            print(f"  Skipping 4 bytes for UINT32")
        elif value_type == 5:  # INT32
            print(f"  Skipping 4 bytes for INT32")
        elif value_type == 6:  # FLOAT32
            print(f"  Skipping 4 bytes for FLOAT32")
        elif value_type == 7:  # BOOL
            print(f"  Skipping 1 byte for BOOL")
        elif value_type == 8:  # STRING
            str_len_bytes = f.read(8)
            str_len = struct.unpack('<Q', str_len_bytes)[0]
            print(f"  string len: {str_len}")
            str_data = f.read(str_len)
            print(f"  string value: {str_data}")
        elif value_type == 9:  # ARRAY
            print(f"  Reading array...")
        else:
            print(f"  Unknown type {value_type}, stopping")
            break

    # Check where we are now
    print(f"\n=== After parsing 5 metadata entries, at pos {f.tell()} ===")

    # Read first tensor info
    print("\n=== Reading first tensor info ===")
    tensor_pos = f.tell()
    print(f"Tensor info starts at pos {tensor_pos}")

    # Read name as uint64 string
    name_len_bytes = f.read(8)
    name_len = struct.unpack('<Q', name_len_bytes)[0]
    print(f"name_len (uint64): {name_len}")

    if name_len > 1000:
        # Try uint32 instead
        f.seek(tensor_pos)
        name_len_bytes = f.read(4)
        name_len = struct.unpack('<I', name_len_bytes)[0]
        print(f"name_len (uint32): {name_len}")
        f.seek(tensor_pos + 4)

    name = f.read(name_len)
    print(f"name: {name}")

    # Read n_dims
    n_dims_bytes = f.read(4)
    n_dims = struct.unpack('<I', n_dims_bytes)[0]
    print(f"n_dims: {n_dims}")

    # Read dimensions
    for d in range(n_dims):
        dim_bytes = f.read(8)
        dim = struct.unpack('<Q', dim_bytes)[0]
        print(f"  dim[{d}]: {dim}")

    # Read dtype
    dtype_bytes = f.read(4)
    dtype = struct.unpack('<I', dtype_bytes)[0]
    print(f"dtype: {dtype}")

    # Read offset
    offset_bytes = f.read(8)
    offset = struct.unpack('<Q', offset_bytes)[0]
    print(f"offset: {offset}")

    print(f"\n=== Byte dump at tensor_info_start (52) ===")
    f.seek(52)
    data = f.read(32)
    print(f"Bytes 52-83: {data.hex(' ')}")
    print(f"Bytes as uint64 at 52: {struct.unpack('<Q', data[:8])[0]}")
    print(f"Bytes as uint64 at 56: {struct.unpack('<Q', data[4:12])[0]}")
    print(f"Bytes as uint32 at 52: {struct.unpack('<I', data[:4])[0]}")
    print(f"Bytes as uint32 at 56: {struct.unpack('<I', data[4:8])[0]}")
    print(f"Bytes as uint32 at 60: {struct.unpack('<I', data[8:12])[0]}")
    print(f"String at 60 (8 bytes): {data[8:16]}")