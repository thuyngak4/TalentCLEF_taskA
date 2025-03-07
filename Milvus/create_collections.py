from pymilvus import FieldSchema, CollectionSchema, DataType, Collection, utility
from connect_milvus import connect_milvus  # Import kết nối

def create_collection(collection_name):
    connect_milvus()  # Kết nối trước

    # Xóa collection nếu đã tồn tại
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
        print(f"🗑️ Đã xóa collection cũ: {collection_name}")

    # Định nghĩa schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="c_id", dtype=DataType.INT16, max_length=100)
    ]
    schema = CollectionSchema(fields, description="Lưu embeddings")

    # Tạo collection
    collection = Collection(name=collection_name, schema=schema)
    collection.create_index(
        field_name="embedding",
        index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}}
    )
    print("✅ Index đã được tạo!")

    # Load collection để index hoạt động
    collection.load()
    print(f"✅ Collection '{collection_name}' đã được tạo!")

if __name__ == "__main__":
    #collection_name = "job_embeddings" #Embedding của tiếng anh
    #collection_name = "job_embeddings_ger"
    #collection_name = "job_embeddings_span"
    collection_name = "job_embeddings_china"

    create_collection(collection_name)
