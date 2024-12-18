import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Du lieu huan luyen co ban
data = [
    ("Chào bạn", "Chào bạn, tôi có thể giúp gì cho bạn?"),
    ("Bạn là ai?", "Tôi là một chatbot đơn giản."),
    ("Thời tiết hôm nay thế nào?", "Xin lỗi, tôi không thể cung cấp thông tin thời tiết."),
    ("Cảm ơn bạn", "Cảm ơn bạn, chúc bạn một ngày tốt lành!"),
    ("Tạm biệt", "Tạm biệt, hẹn gặp lại!"),
]

# Tach cau hoi va cau tra loi

questions = [item[0] for item in data]
answers = [item[1] for item in data]

# Tao pipeline voi TF-IDF va Naive Bayes
model = make_pipeline(TfidfVectorizer(),MultinomialNB())

# Huan luyen mo hinh
model.fit(questions, answers)

# Tinh ma tran TF-IDF cho du lieu
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)
words = vectorizer.get_feature_names_out()
vectors = X.toarray()

print("Các từ trong bộ dữ liệu:")
print(words)
print("\nMa trận TF-IDF của các câu hỏi:")
print(vectors)
# Ham tra loi cau hoi
def chatbot_respone(user_input):
    respone = model.predict([user_input])
    return respone[0]

# Tao vong lap chat bot
print("Chào bạn! Hãy hỏi tôi một câu hỏi.")
while True:
    user_input = input("Bạn: ")
    if user_input.lower() in ["tạm biệt", "bye", "thôi", "đi"]:
        print("Chatbot: Tạm biệt, hẹn gặp lại!")
        break
    respone = chatbot_respone(user_input)
    print(f"Chatbot: {respone}")