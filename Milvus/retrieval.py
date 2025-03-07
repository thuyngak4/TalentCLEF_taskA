import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from pymilvus import Collection
from connect_milvus import connect_milvus  # Import kết nối
from embedding_jobtitle import average_pool
import pandas as pd
import os

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')

# Tìm kiếm trong Milvus
def search_milvus(collection, query_text, top_k=10):

    # Lấy embedding của query
    batch_dict = tokenizer([query_text], max_length=512, padding=True, truncation=True, return_tensors='pt')

    outputs = model(**batch_dict)
    query_embedding = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    query_embedding = F.normalize(query_embedding, p=2, dim=1).tolist()
    query_embedding = query_embedding[0] 
    # Thực hiện tìm kiếm
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_embedding], 
        anns_field="embedding", 
        param=search_params, 
        limit=top_k, 
        output_fields=["c_id"]
    )

    return results[0]  # Trả về danh sách tài liệu phù hợp

# Xuất file theo TREC Run Format
def generate_trec_run(queries_df, output_file="trec_run_china.trec", top_k=10):
    connect_milvus()
    collection = Collection("job_embeddings_china")
    with open(output_file, "w") as f:
        for q_id, query_text in zip(queries_df.q_id, queries_df.jobtitle):
            results = search_milvus(collection, query_text, top_k=top_k)
            for rank, hit in enumerate(results):
                doc_id = hit.entity.get("c_id")  # ID của tài liệu trong corpus
                score = float(hit.distance)   # Điểm số tương tự
                f.write(f"{q_id} Q0 {doc_id+1} {rank+1} {score:.4f} my_experiment\n")

    print(f"✅ File {output_file} đã được tạo thành công!")

queries_path = os.getenv("file_path_val_chi_query")

queries_df = pd.read_csv(queries_path, sep="\t")

# Tạo file TREC Run
generate_trec_run(queries_df, output_file="trec_results_china.trec", top_k=20) 