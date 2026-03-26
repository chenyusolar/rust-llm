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
    
    print(f"\nAfter header (pos 24):")
    pos24 = f.tell()
    data24 = f.read(32)
    print(f"Bytes 24-55: {data24.hex(' ')}")
    
    # Check bytes 16-23
    f.seek(16)
    data16 = f.read(8)
    print(f"\nBytes 16-23: {data16.hex(' ')}")
    print(f"As uint64: {struct.unpack('<Q', data16)[0]}")
    
    # If metadata_count is wrong, try different interpretation
    print(f"\n--- Alternative interpretation ---")
    print(f"If metadata_count is actually at bytes 12-15 (uint32):")
    f.seek(12)
    mc32 = struct.unpack('<I', f.read(4))[0]
    print(f"Metadata count (uint32): {mc32}")
    
    print(f"\nBytes 20-23 (might be tensor_info_offset):")
    f.seek(20)
    data20 = f.read(8)
    print(f"Bytes 20-27: {data20.hex(' ')}")
    print(f"As uint64: {struct.unpack('<Q', data20)[0]}")
    
    # Check actual tensor info position
    print(f"\n--- Checking tensor info at various positions ---")
    f.seek(24)
    key_len = struct.unpack('<Q', f.read(8))[0]
    print(f"Key len at pos 24 (uint64): {key_len}")
    
    key_data = f.read(key_len)
    print(f"Key data at pos 32: {key_data}")
    
    value_type = struct.unpack('<I', f.read(4))[0]
    print(f"Value type at pos 40: {value_type}")
    
    # If value_type is 6 (F32), read 4 bytes
    if value_type == 6:
        value_data = f.read(4)
        print(f"Value data (F32) at pos 44: {value_data.hex(' ')}")
        metadata_end = 48
    else:
        metadata_end = 44
    
    print(f"\nAssuming metadata ends at byte {metadata_end}")
    print(f"Tensor info should start at byte {metadata_end}")
    
    # Check tensor info at that position
    f.seek(metadata_end)
    ti_data = f.read(32)
    print(f"Bytes at {metadata_end}: {ti_data.hex(' ')}")
    
    # Try reading tensor info as uint32 name_len
    f.seek(metadata_end)
    name_len_u32 = struct.unpack('<I', f.read(4))[0]
    print(f"\nName len at {metadata_end} (uint32): {name_len_u32}")
    
    name_data = f.read(name_len_u32)
    print(f"Name data: {name_data}")
