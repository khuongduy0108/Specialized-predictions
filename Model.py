import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Đọc dữ liệu từ tập tin Excel
data = pd.read_excel("D:/DATN/App_Goi_Y/Data/Data_Total.xlsx")

# Xác định các trường đầu vào và nhãn
features = data.drop("Labels", axis=1)
labels = data["Labels"]

# Chuyển đổi nhãn về dạng số
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Huấn luyện mô hình SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Dự đoán nhãn trên tập kiểm tra
y_pred = svm_model.predict(X_test)

# Đánh giá độ chính xác của mô hình
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
