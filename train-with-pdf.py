import numpy as np
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber


def extract_text_from_pdf(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"  # Trích xuất văn bản từ mỗi trang
            return text
    except Exception as e:
        print(f"Lỗi khi đọc PDF: {e}")
        return ""

# Chuan bi du lieu
def prepare_knowlegde_base(text):
    sentences = text.split(".")
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


# Tim kiem cau tra loi
def find_best_answer(question, sentences, vectorizer):
    # Vecto hoa cau hoi va du lieu
    vectors = vectorizer.transform(sentences)
    question_vector = vectorizer.transform([question])

    # Tinh toan do tuong dong cosline
    similarities = cosine_similarity(question_vector, vectors).flatten()
    best_index = np.argmax(similarities)

    return sentences[best_index], similarities[best_index]


# Doc tai lieu nay
load_dotenv()
pdf_path = os.getenv("PDF_PATH")
document_text = extract_text_from_pdf(pdf_path)
knowledge_base = prepare_knowlegde_base(document_text)

# Vectorizer voi TF-IDF
vectorizer = TfidfVectorizer().fit(knowledge_base)

# Check loi voi knowledge_base
print(f"knowlegde_base: {knowledge_base}")
# Bat dau chatbot
print("Chatbot: Tôi đã đọc tài liệu, hãy hỏi tôi điều gì đó!")
while True:
    user_input = input("Bạn: ")
    if user_input.lower() in ["tạm biệt", "bye", "exit"]:
        print("Chatbot: Tạm biệt, hẹn gặp lại!")
        break
    # Tim cau tra loi tot nhat
    answer, similarities = find_best_answer(user_input, knowledge_base, vectorizer)
    if similarities > 0.5:
        print(f"Chatbot: {answer}")
    else:
        print("Chatbot: Xin lỗi, tôi không tìm thấy câu trả lời phù hợp.")
