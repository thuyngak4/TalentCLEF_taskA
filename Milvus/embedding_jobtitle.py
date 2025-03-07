import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from pymilvus import Collection
from connect_milvus import connect_milvus  # Import kết nối
import pandas as pd
import os

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')

# Hàm tính trung bình pooling
def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def insert_embeddings(collection_name, input_texts, input_ids):
    connect_milvus()  # Kết nối Milvus
    collection = Collection(collection_name)  # Mở collection

    batch_size = 200  # Số lượng phần tử mỗi batch
    num_batches = (len(input_texts) + batch_size - 1) // batch_size  # Tính số batch

    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(input_texts))
        batch_texts = input_texts[start:end]

        # Tokenize & lấy embeddings
        batch_dict = tokenizer(batch_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
        outputs = model(**batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1).tolist()  # Chuyển thành list

        print(f"Batch {i+1}/{num_batches} - Shape: {len(embeddings)} x {len(embeddings[0])}")

        # Insert vào Milvus
        input_ids_batch = list(range(start, end))  # Tạo danh sách id tương ứng
        collection.insert([embeddings, input_ids_batch])
        collection.flush()

    print("✅ Tất cả embeddings đã được lưu vào Milvus!")

if __name__ == "__main__":
    # insert_embeddings()
    file_path = os.getenv("file_path_val_chi_corpus")
    queries_df = pd.read_csv(file_path, sep="\t")

    #collection_name = "job_embeddings" #Embedding của tiếng anh
    #collection_name = "job_embeddings_ger"
    #collection_name = "job_embeddings_span"
    collection_name = "job_embeddings_china"

    insert_embeddings(collection_name, queries_df.jobtitle.to_list(), queries_df.c_id.to_list())
    
    

