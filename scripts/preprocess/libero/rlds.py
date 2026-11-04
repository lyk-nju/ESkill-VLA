"""Read TFDS Example records without loading TensorFlow or a policy model."""
import struct

import google_crc32c
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


def example_type():
    schema = descriptor_pb2.FileDescriptorProto(name="raw_example.proto", package="raw_check", syntax="proto3")
    for name, field_type in [("BytesList", 12), ("FloatList", 2), ("Int64List", 3)]:
        message = schema.message_type.add(name=name)
        message.field.add(name="value", number=1, type=field_type, label=3)
    feature = schema.message_type.add(name="Feature")
    feature.oneof_decl.add(name="kind")
    for index, name in enumerate(["BytesList", "FloatList", "Int64List"], 1):
        field = feature.field.add(name=["bytes_list", "float_list", "int64_list"][index-1],
                                  number=index, type=11, label=1,
                                  type_name=".raw_check." + name)
        field.oneof_index = 0
    features = schema.message_type.add(name="Features")
    entry = features.nested_type.add(name="FeatureEntry")
    entry.options.map_entry = True
    entry.field.add(name="key", number=1, type=9, label=1)
    entry.field.add(name="value", number=2, type=11, label=1, type_name=".raw_check.Feature")
    features.field.add(name="feature", number=1, type=11, label=3, type_name=".raw_check.Features.FeatureEntry")
    example = schema.message_type.add(name="Example")
    example.field.add(name="features", number=1, type=11, label=1, type_name=".raw_check.Features")
    pool = descriptor_pool.DescriptorPool()
    pool.Add(schema)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("raw_check.Example"))


Example = example_type()


def masked_crc(data):
    crc = google_crc32c.value(data)
    return (((crc >> 15) | (crc << 17)) + 0xa282ead8) & 0xffffffff


def _record(stream, path):
    """Decode the record at the stream position; verify both TFRecord CRCs. None at end of file."""
    offset = stream.tell()
    header = stream.read(12)
    if not header:
        return None
    if len(header) != 12 or masked_crc(header[:8]) != struct.unpack("<I", header[8:])[0]:
        raise ValueError(f"Invalid record header at {path}:{offset}")
    size = struct.unpack("<Q", header[:8])[0]
    data = stream.read(size)
    tail = stream.read(4)
    if len(data) != size or len(tail) != 4 or masked_crc(data) != struct.unpack("<I", tail)[0]:
        raise ValueError(f"Invalid record body at {path}:{offset}")
    example = Example.FromString(data)
    decoded = {}
    for key, feature in example.features.feature.items():
        kind = feature.WhichOneof("kind")
        decoded[key] = list(getattr(feature, kind).value)
    return decoded


def records(path):
    """Yield index, byte offset and decoded feature values."""
    with open(path, "rb") as stream:
        index = 0
        while True:
            offset = stream.tell()
            decoded = _record(stream, path)
            if decoded is None:
                return
            yield index, offset, decoded
            index += 1


def record_at(path, offset):
    """One record at a known byte offset, without scanning the shard."""
    with open(path, "rb") as stream:
        stream.seek(offset)
        decoded = _record(stream, path)
    if decoded is None:
        raise ValueError(f"No record at {path}:{offset}")
    return decoded


def load_records(root, samples):
    """Read selected complete episodes, grouped by shard to avoid repeated scans."""
    for shard in sorted({sample['shard'] for sample in samples}):
        chosen = {sample['record_index']: sample for sample in samples if sample['shard'] == shard}
        for row, _, data in records(root / shard):
            if row in chosen:
                yield chosen[row], data
