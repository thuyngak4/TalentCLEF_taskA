from pymilvus import connections
import os
from dotenv import load_dotenv

load_dotenv()

# Thông tin kết nối
URI = os.getenv("endpoint_milvus")  # Public Endpoint từ Zilliz
TOKEN = os.getenv("token")  # Token từ Zilliz

def connect_milvus():
    connections.connect(alias="default", uri=URI, token=TOKEN)
    print("✅ Kết nối Milvus thành công!")

if __name__ == "__main__":
    connect_milvus()
