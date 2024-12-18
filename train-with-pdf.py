import numpy as np
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
import spacy

nlp = spacy.load("vi_core_news_lg")

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

# Xay dung mo hinh nlp
def chatbot_response(question,knowledge_base):
    question_vector = nlp(question)
    best_match = None
    best_score = 0
    for key,value in knowledge_base.items():
        key_vector = nlp(key)
        similarity = question_vector.similarity(key_vector)
        if similarity > best_score:
            best_match = key
            best_score = similarity
    
    if best_match and best_score > 0.7:  # Ngưỡng tương đồng
        return knowledge_base[best_match]
    else:
        return "Xin lỗi, tôi không tìm thấy câu trả lời phù hợp."
# Bat dau chatbot
print("Chatbot: Tôi đã đọc tài liệu, hãy hỏi tôi điều gì đó!")
while True:
    user_input = input("Bạn: ")
    if user_input.lower() in ["exit", "thoát"]:
        print("Chatbot: Tạm biệt!")
        break
    response = chatbot_response(user_input, knowledge_base)
    print(f"Chatbot: {response}")