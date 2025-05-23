# Amazon Reviews Üzerinde Sentiment Analysis, Topic Modeling ve Dinamik Soru–Cevap Sistemi

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# NLTK paketlerini indir
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# 1) Veri Hazırlığı ve Temizleme
# -----------------------------
df = pd.read_csv('amazon_review.csv')
df = df[['reviewText', 'overall']].dropna()
df = df[df['overall'] != 3]  # 3 yıldızlıları çıkar
df['sentiment'] = df['overall'].apply(lambda x: 1 if x >= 4 else 0)

# Metin Ön İşleme Fonksiyonu
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stop_words]
    return ' '.join(tokens)

# Temizlenmiş metin sütunu
df['clean_text'] = df['reviewText'].apply(preprocess_text)
texts = df['clean_text'].tolist()

# 2) TF-IDF Vektörleştirme (Sentiment & QA)
# -----------------------------------------
tfidf = TfidfVectorizer(max_df=0.9, min_df=5)
X_tfidf = tfidf.fit_transform(texts)

# 3) Sentiment Sınıflandırma (Voting + Stacking Ensemble ile)
# ---------------------------------------------------------
train_texts, test_texts, y_train, y_test = train_test_split(
    texts, df['sentiment'], test_size=0.2, random_state=42
)
X_train = tfidf.transform(train_texts)
X_test = tfidf.transform(test_texts)

# Bireysel modeller
lr_model = LogisticRegression(max_iter=1000)
nb_model = MultinomialNB()
svm_model = LinearSVC()

# Voting ensemble modeli (hard voting)
voting_model = VotingClassifier(
    estimators=[('lr', lr_model), ('nb', nb_model), ('svm', svm_model)],
    voting='hard'
)

# Stacking ensemble modeli
stacking_model = StackingClassifier(
    estimators=[('lr', lr_model), ('nb', nb_model), ('svm', svm_model)],
    final_estimator=LogisticRegression(max_iter=1000)
)

# Modelleri eğit
voting_model.fit(X_train, y_train)
stacking_model.fit(X_train, y_train)

# Tahmin yap
y_pred_voting = voting_model.predict(X_test)
y_pred_stacking = stacking_model.predict(X_test)

print("--- Sentiment Classification with Voting Ensemble ---")
print("Accuracy:", accuracy_score(y_test, y_pred_voting))
print(classification_report(y_test, y_pred_voting, target_names=['negative', 'positive']))

print("--- Sentiment Classification with Stacking Ensemble ---")
print("Accuracy:", accuracy_score(y_test, y_pred_stacking))
print(classification_report(y_test, y_pred_stacking, target_names=['negative', 'positive']))

# 4) Topic Modeling (LDA)
# -----------------------
num_topics = 5
count_vec = CountVectorizer(max_df=0.95, min_df=5)
dtm = count_vec.fit_transform(texts)
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(dtm)
feature_names = count_vec.get_feature_names_out()

def display_topics(model, feature_names, no_top_words=10):
    for idx, topic in enumerate(model.components_):
        print(f"Topic {idx+1}: ", ", ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

print("\n--- Topics ---")
display_topics(lda, feature_names)

# 5) Ana Program Akışı
# ---------------------
if __name__ == '__main__':
    # Grafiklerin gösterimi
    print("Grafikler oluşturuluyor...")
    plt.figure()
    df['sentiment'].value_counts().sort_index().plot(kind='bar')
    plt.title('Sentiment Distribution (0=Neg, 1=Pos)')
    plt.tight_layout()
    plt.show()

    # Confusion Matrix grafikleri
    cm_voting = confusion_matrix(y_test, y_pred_voting)
    cm_stacking = confusion_matrix(y_test, y_pred_stacking)

    plt.figure()
    disp_voting = ConfusionMatrixDisplay(cm_voting, display_labels=['Neg','Pos'])
    disp_voting.plot()
    plt.title('Confusion Matrix - Voting Ensemble')
    plt.tight_layout()
    plt.show()

    plt.figure()
    disp_stacking = ConfusionMatrixDisplay(cm_stacking, display_labels=['Neg','Pos'])
    disp_stacking.plot()
    plt.title('Confusion Matrix - Stacking Ensemble')
    plt.tight_layout()
    plt.show()

    for idx, topic in enumerate(lda.components_):
        top_ids = topic.argsort()[:-11:-1]
        top_words = [feature_names[i] for i in top_ids]
        weights = topic[top_ids]
        plt.figure()
        plt.bar(range(len(top_words)), weights)
        plt.xticks(range(len(top_words)), top_words, rotation=45, ha='right')
        plt.title(f'Topic {idx+1} Top Words')
        plt.tight_layout()
        plt.show()

    # 6) Dinamik Soru–Cevap Sistemi
    print("\nDinamik Soru–Cevap Sistemi (çıkmak için 'quit' yazın)")
    used_idxs = set()
    while True:
        query = input("Soru: ").strip()
        if query.lower() == 'quit':
            print("Programdan çıkılıyor...")
            break
        if not query:
            print("Lütfen geçerli bir soru girin veya 'quit' ile çıkın.")
            continue
        # Soruyu temizle ve vektörleştir
        q_clean = preprocess_text(query)
        q_vec = tfidf.transform([q_clean])
        sims = cosine_similarity(q_vec, X_tfidf).flatten()
        sorted_idxs = sims.argsort()[::-1]
        # Yanıt bul
        found = False
        for idx in sorted_idxs:
            if idx not in used_idxs:
                used_idxs.add(idx)
                score = sims[idx]
                print(f"  - Cevap (score={score:.2f}): {df['reviewText'].iloc[idx]}")
                found = True
                break
        if not found:
            print("Uygun yeni cevap bulunamadı.")
