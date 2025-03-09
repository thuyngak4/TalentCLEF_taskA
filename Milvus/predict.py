from transformers import BertTokenizer, BertForSequenceClassification
import torch

def predict(job1, job2,  max_length=400):
# Load tokenizer và model từ thư mục đã lưu
    tokenizer = BertTokenizer.from_pretrained(r"D:\UIT\TaskA_job\finetuning_Bert\saved_model")
    model = BertForSequenceClassification.from_pretrained(r"D:\UIT\TaskA_job\finetuning_Bert\saved_model")

    # Chuyển model về chế độ eval để dự đoán
    model.eval()

    """Dự đoán nhãn (0 hoặc 1) cho cặp job titles"""
    
    # Tokenize dữ liệu đầu vào
    encoding = tokenizer(
        job1, job2,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    # Đưa dữ liệu vào model để dự đoán
    with torch.no_grad():  # Không tính toán gradient để tiết kiệm bộ nhớ
        outputs = model(**encoding)
    
    logits = outputs.logits  # Lấy giá trị logits từ model
    predicted_class = torch.argmax(logits, dim=1).item()  # Chọn class có xác suất cao nhất (0 hoặc 1)
    
    return predicted_class
# job1 = "nanny"
# job2 = "daycare teacher"

# label = predict(job1, job2, model, tokenizer)
# print(f"Predicted label: {label}")

# with open(r"TaskA\validation\english\corpus_elements", "r", encoding="utf-8") as f:
#     corpus = [line.strip() for line in f]

# # Từ gán để so sánh
# a = "nanny"

# # Duyệt từng từ trong corpus, dự đoán và in nếu label = 1
# for word in corpus:
#     label = predict(a, word, model, tokenizer)
#     if label == 1:
#         print(word)