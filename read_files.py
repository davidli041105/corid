import pyarrow as pa
import pyarrow.ipc as ipc

with pa.memory_map(
    "data-00000-of-00001.arrow",
    "r"
) as source:

    table = ipc.open_stream(source).read_all()

print(table.schema)
print()
print(table.to_pandas().head())