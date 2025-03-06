import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection

# Thay thế bằng thông tin của bạn từ Zilliz Cloud
URI = "https://your-zilliz-uri"
TOKEN = "your-token"

# Kết nối tới Milvus
connections.connect(alias="default", uri=URI, token=TOKEN)

# Kiểm tra kết nối
print("Kết nối thành công:", utility.list_collections())

# Định nghĩa schema cho collection
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)  # Thay 768 bằng kích thước embedding của bạn
]

schema = CollectionSchema(fields, description="Lưu embeddings từ mô hình")

# Tạo collection
collection_name = "my_embeddings"
collection = Collection(name=collection_name, schema=schema)

# Index để tìm kiếm nhanh hơn
index_params = {
    "metric_type": "COSINE",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128}
}
collection.create_index(field_name="embedding", index_params=index_params)

print(f"Collection {collection_name} đã được tạo thành công!")


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')

