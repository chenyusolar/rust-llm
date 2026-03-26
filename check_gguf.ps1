$ErrorActionPreference = 'SilentlyContinue'
$file = [System.IO.File]::OpenRead('e:\models\qwen\Qwen3.5-9B-Q4_K_M.gguf')
$br = New-Object System.IO.BinaryReader($file)

$magic = $br.ReadBytes(4)
Write-Host "Magic: $([char[]]($magic))"

$version = $br.ReadUInt32()
Write-Host "Version: $version"

$tensorCount = $br.ReadUInt64()
Write-Host "Tensor count: $tensorCount"

$metadataCount = $br.ReadUInt64()
Write-Host "Metadata count: $metadataCount"

$pos = $br.BaseStream.Position
Write-Host "`nAfter header, position: $pos"
$bytes = $br.ReadBytes(40)
Write-Host "Bytes $pos-$($pos+39): $(($bytes | ForEach-Object { '{0:X2}' -f $_ }) -join ' ')"

# Check bytes 16-23
$file.Seek(16, [System.IO.SeekOrigin]::Begin) | Out-Null
$bytes16 = $file.ReadBytes(8)
Write-Host "`nBytes 16-23: $(($bytes16 | ForEach-Object { '{0:X2}' -f $_ }) -join ' ')"

# Try uint32 interpretation
$file.Seek(12, [System.IO.SeekOrigin]::Begin) | Out-Null
$mc32 = $br.ReadUInt32()
Write-Host "Bytes 12-15 (uint32): $mc32"

# Check bytes 20-23
$file.Seek(20, [System.IO.SeekOrigin]::Begin) | Out-Null
$bytes20 = $file.ReadBytes(8)
Write-Host "Bytes 20-27: $(($bytes20 | ForEach-Object { '{0:X2}' -f $_ }) -join ' ')"

# Metadata analysis
$file.Seek(24, [System.IO.SeekOrigin]::Begin) | Out-Null
$keyLen = $br.ReadUInt64()
Write-Host "`nKey len at pos 24 (uint64): $keyLen"
$keyData = $br.ReadBytes($keyLen)
Write-Host "Key data: $([System.Text.Encoding]::ASCII.GetString($keyData))"

$valueType = $br.ReadUInt32()
Write-Host "Value type at pos 32: $valueType"

$file.Close()
