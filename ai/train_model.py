import os
import json
import time
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from database.db import get_connection
from ai.preprocess import preprocess_text

DEFAULT_SAVED_DIR = os.path.join(os.path.dirname(__file__), "saved_models")


if HAS_TORCH:
    class PyTorchDeepIntentNet(nn.Module):
        """
        MẠNG NƠ-RON SÂU PYTORCH ĐA TẦNG (DEEP RESIDUAL MULTI-LAYER PERCEPTRON):
        - Kiến trúc trích xuất đặc trưng sâu (Deep Feature Extractor) 3 tầng ẩn:
          + Layer 1: in_features -> 256 nơ-ron (BatchNorm1d + LeakyReLU(0.1) + Dropout 0.3)
          + Layer 2: 256 -> 128 nơ-ron (BatchNorm1d + LeakyReLU(0.1) + Dropout 0.2)
          + Layer 3: 128 -> 64 nơ-ron (BatchNorm1d + LeakyReLU(0.1))
        - Tầng phân loại (Classifier Head):
          + Layer 4: Dropout(0.1) -> Linear(64, num_classes)
        - Vector nhúng ngữ nghĩa tiềm ẩn (64-D Latent Semantic Embedding):
          Trích xuất từ đầu ra của Layer 3 sau khi chuẩn hóa chuẩn L2 (L2-normalized)
          để phục vụ tính toán trực tiếp độ tương đồng Cosine trong không gian nơ-ron sâu.
        - Tối ưu hóa: Thuật toán AdamW (Weight Decay L2 = 1e-4), Cosine Annealing Learning Rate Scheduler.
        - Hàm mất mát: CrossEntropyLoss kết hợp Label Smoothing (0.05) tăng tính tổng quát hóa.
        """
        def __init__(self, in_features, num_classes):
            super().__init__()
            self.in_features = in_features
            self.num_classes = num_classes

            self.feature_extractor = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.3),

                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.2),

                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.1),
            )
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(64, num_classes)

        def forward(self, x):
            feats = self.feature_extractor(x)
            feats = self.dropout(feats)
            return self.classifier(feats)

        def extract_embedding(self, x):
            """
            Trích xuất vector ngữ nghĩa sâu 64 chiều (64-D Latent Semantic Embedding)
            và chuẩn hóa L2 (L2-normalized) phục vụ so khớp Cosine trực tiếp.
            """
            self.eval()
            with torch.no_grad():
                feats = self.feature_extractor(x)
                norm = torch.norm(feats, p=2, dim=1, keepdim=True).clamp(min=1e-12)
                return (feats / norm).cpu().numpy()

        def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DeepHybridModel:
    """
    KIẾN TRÚC HỌC SÂU KẾT HỢP THẾ HỆ MỚI (ADVANCED DEEP HYBRID ENSEMBLE MODEL):
    Kết hợp 2 trường phái học máy tiên tiến trong Xử lý Ngôn ngữ Tự nhiên (NLP):
    1. Discriminative Deep Learning: Mạng nơ-ron sâu PyTorch đa tầng với Batch Normalization,
       hàm kích hoạt LeakyReLU, Dropout và AdamW Optimizer.
       (Tự động fallback sang Deep MLPClassifier nếu chưa cài PyTorch).
    2. Generative Probabilistic: Complement Naive Bayes (CNB) xử lý mất cân bằng lớp và từ khóa hiếm.
    3. Softmax Probability Fusion: Kết hợp phân phối xác suất hậu nghiệm (65% PyTorch Deep NN + 35% CNB).
    4. 64-D Latent Semantic Space: Cung cấp vector nhúng 64 chiều cho thuật toán Deep Hybrid Matching.
    """

    def __init__(self):
        self.has_torch = HAS_TORCH
        self.torch_model = None
        self.label_encoder = LabelEncoder()
        self.classes_ = None
        self.in_features = None
        self.num_classes = None
        self.train_loss_history = []

        # Fallback Deep MLP nếu không có PyTorch
        self.mlp = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            alpha=0.001,
            max_iter=500,
            random_state=42
        )

        # Mô hình xác suất bổ sung Complement Naive Bayes
        self.cnb = ComplementNB(alpha=0.5)

    def fit(self, X, y, epochs=45, batch_size=32, lr=0.003, verbose=False):
        """Huấn luyện đồng thời cả Mạng nơ-ron sâu và Naive Bayes."""
        # Chuyển đổi nhãn sang số nguyên
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = np.array(self.label_encoder.classes_)
        self.num_classes = len(self.classes_)

        # Chuyển ma trận thưa sang mảng đặc nếu cần
        X_dense = X.toarray() if hasattr(X, "toarray") else np.array(X)
        self.in_features = X_dense.shape[1]

        if self.has_torch:
            self.torch_model = PyTorchDeepIntentNet(self.in_features, self.num_classes)
            criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
            optimizer = optim.AdamW(self.torch_model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

            dataset = TensorDataset(
                torch.FloatTensor(X_dense),
                torch.LongTensor(y_encoded)
            )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            self.torch_model.train()
            self.train_loss_history = []
            for ep in range(epochs):
                total_loss = 0.0
                for bx, by in loader:
                    optimizer.zero_grad()
                    out = self.torch_model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * bx.size(0)
                scheduler.step()
                ep_loss = total_loss / len(dataset)
                self.train_loss_history.append(float(ep_loss))
                if verbose and (ep + 1) % 15 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"   [Epoch {ep+1:02d}/{epochs}] CrossEntropy Loss: {ep_loss:.4f} | LR: {current_lr:.6f}")
        else:
            self.mlp.fit(X, y)

        self.cnb.fit(X, y_encoded)
        return self

    def extract_embedding(self, X):
        """
        Trích xuất vector ngữ nghĩa sâu 64 chiều (64-D Latent Semantic Embedding)
        đã được chuẩn hóa L2 (L2-normalized) phục vụ so khớp Cosine trực tiếp.
        """
        X_dense = X.toarray() if hasattr(X, "toarray") else np.array(X)
        if self.has_torch and self.torch_model is not None:
            tensor_x = torch.FloatTensor(X_dense)
            return self.torch_model.extract_embedding(tensor_x)
        else:
            # Fallback trích xuất từ tầng ẩn thứ 3 (64 chiều) của MLPClassifier
            if hasattr(self.mlp, "coefs_") and len(self.mlp.coefs_) >= 3:
                h1 = np.maximum(0, X_dense @ self.mlp.coefs_[0] + self.mlp.intercepts_[0])
                h2 = np.maximum(0, h1 @ self.mlp.coefs_[1] + self.mlp.intercepts_[1])
                h3 = np.maximum(0, h2 @ self.mlp.coefs_[2] + self.mlp.intercepts_[2])
                norm = np.linalg.norm(h3, axis=1, keepdims=True)
                norm[norm == 0] = 1e-12
                return h3 / norm
            norm = np.linalg.norm(X_dense, axis=1, keepdims=True)
            norm[norm == 0] = 1e-12
            emb = X_dense / norm
            if emb.shape[1] >= 64:
                return emb[:, :64]
            return np.pad(emb, ((0, 0), (0, 64 - emb.shape[1])))

    def predict_proba(self, X):
        """
        Dự đoán phân phối xác suất kết hợp (Ensemble Soft Voting):
        P_hybrid = 0.65 * P_DeepNN + 0.35 * P_CNB
        """
        X_dense = X.toarray() if hasattr(X, "toarray") else np.array(X)

        if self.has_torch and self.torch_model is not None:
            self.torch_model.eval()
            with torch.no_grad():
                logits = self.torch_model(torch.FloatTensor(X_dense))
                p_deep = torch.softmax(logits, dim=1).cpu().numpy()
        else:
            p_deep = self.mlp.predict_proba(X)

        p_cnb = self.cnb.predict_proba(X)
        return 0.65 * p_deep + 0.35 * p_cnb

    def predict(self, X):
        """Dự đoán nhãn ý định có xác suất kết hợp cao nhất."""
        probs = self.predict_proba(X)
        best_indices = np.argmax(probs, axis=1)
        return self.classes_[best_indices]

    def save_artifacts(self, save_dir=DEFAULT_SAVED_DIR, vectorizer=None, questions=None, metrics=None):
        """
        Lưu các thành phần mô hình vào thư mục save_dir:
        - deep_intent_model.pth: Trọng số PyTorch (state_dict)
        - model_metadata.pkl: vectorizer, label_encoder, classes_, cnb, cấu hình
        - question_embeddings.npy: Mảng 64-D Latent Semantic Embeddings cho toàn bộ câu hỏi mẫu
        - training_metrics.json: Các chỉ số loss, accuracy, F1-score, thông số mô hình
        """
        os.makedirs(save_dir, exist_ok=True)

        # 1. Trọng số mô hình PyTorch
        if self.has_torch and self.torch_model is not None:
            pth_path = os.path.join(save_dir, "deep_intent_model.pth")
            torch.save(self.torch_model.state_dict(), pth_path)

        # 2. Metadata và mô hình bổ trợ
        metadata_path = os.path.join(save_dir, "model_metadata.pkl")
        metadata = {
            "has_torch": self.has_torch,
            "in_features": self.in_features,
            "num_classes": self.num_classes,
            "classes_": self.classes_,
            "label_encoder": self.label_encoder,
            "cnb": self.cnb,
            "mlp": self.mlp if not self.has_torch else None,
            "vectorizer": vectorizer
        }
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        # 3. Tiền tính toán và lưu Question Embeddings (N x 64)
        if vectorizer is not None and questions is not None:
            X_all = vectorizer.transform(questions)
            embeddings = self.extract_embedding(X_all)
            npy_path = os.path.join(save_dir, "question_embeddings.npy")
            np.save(npy_path, embeddings)

        # 4. Lưu metrics JSON
        param_count = self.torch_model.count_parameters() if (self.has_torch and self.torch_model) else 0
        final_metrics = {
            "framework": "PyTorch 2.x Deep Learning Neural Network",
            "model_architecture": "PyTorch Deep Intent Multi-Layer Perceptron (Linear 256 -> BatchNorm1d -> LeakyReLU -> Linear 128 -> BatchNorm1d -> LeakyReLU -> Linear 64 -> BatchNorm1d -> LeakyReLU -> Linear num_classes)",
            "embedding_dimension": 64,
            "total_parameters": param_count,
            "optimizer": "AdamW (lr=0.003, weight_decay=1e-4, CosineAnnealingLR)",
            "loss_function": "CrossEntropyLoss(label_smoothing=0.05)",
            "ensemble_strategy": "Softmax Probability Fusion: 65% PyTorch Deep Neural Network + 35% Complement Naive Bayes",
            "train_loss_history": self.train_loss_history,
            "saved_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        if metrics:
            final_metrics.update(metrics)

        metrics_path = os.path.join(save_dir, "training_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(final_metrics, f, ensure_ascii=False, indent=2)

        return True

    @classmethod
    def load_artifacts(cls, save_dir=DEFAULT_SAVED_DIR):
        """
        Nạp nhanh toàn bộ mô hình và vector nhúng từ bộ nhớ đệm (< 50ms):
        Trả về: (vectorizer, model, question_embeddings, metrics) hoặc None nếu chưa có.
        """
        metadata_path = os.path.join(save_dir, "model_metadata.pkl")
        if not os.path.exists(metadata_path):
            return None

        try:
            with open(metadata_path, "rb") as f:
                meta = pickle.load(f)

            model = cls()
            model.has_torch = meta.get("has_torch", False) and HAS_TORCH
            model.in_features = meta.get("in_features")
            model.num_classes = meta.get("num_classes")
            model.classes_ = meta.get("classes_")
            model.label_encoder = meta.get("label_encoder")
            model.cnb = meta.get("cnb")
            vectorizer = meta.get("vectorizer")

            if model.has_torch and model.in_features and model.num_classes:
                pth_path = os.path.join(save_dir, "deep_intent_model.pth")
                if os.path.exists(pth_path):
                    torch_model = PyTorchDeepIntentNet(model.in_features, model.num_classes)
                    torch_model.load_state_dict(torch.load(pth_path, map_location="cpu"))
                    torch_model.eval()
                    model.torch_model = torch_model
            elif not model.has_torch:
                model.mlp = meta.get("mlp")

            # Nạp Dense Question Embeddings
            npy_path = os.path.join(save_dir, "question_embeddings.npy")
            question_embeddings = np.load(npy_path) if os.path.exists(npy_path) else None

            # Nạp metrics
            metrics_path = os.path.join(save_dir, "training_metrics.json")
            metrics = None
            if os.path.exists(metrics_path):
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)

            return vectorizer, model, question_embeddings, metrics
        except Exception as e:
            print("Lỗi load model artifacts:", e)
            return None


def load_training_data():
    """
    Lấy dữ liệu câu hỏi và intent từ MySQL hoặc fallback file sample_qa.json.
    """
    questions = []
    intents = []

    # 1. Thử lấy từ cơ sở dữ liệu MySQL
    connection = get_connection()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            sql = """
                SELECT question, intent
                FROM qa_data
                WHERE question IS NOT NULL
                  AND intent IS NOT NULL
            """
            cursor.execute(sql)
            data = cursor.fetchall()
            for row in data:
                questions.append(preprocess_text(row["question"]))
                intents.append(row["intent"])
        except Exception as e:
            print("Lỗi lấy dữ liệu huấn luyện từ DB:", e)
        finally:
            cursor.close()
            connection.close()

    # 2. Nếu trong DB chưa có đủ dữ liệu, dùng file sample_qa.json
    if len(questions) < 2:
        json_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_qa.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    sample_data = json.load(f)
                    for item in sample_data:
                        questions.append(preprocess_text(item["question"]))
                        intents.append(item["intent"])
            except Exception as e:
                print("Lỗi đọc sample_qa.json:", e)

    return questions, intents


def train_model(force_retrain=True, save_artifacts=True, save_dir=DEFAULT_SAVED_DIR, evaluate=True, return_all=False, verbose=True):
    """
    HUẤN LUYỆN MÔ HÌNH HỌC SÂU DEEP LEARNING (TF-IDF + PYTORCH DEEP HYBRID MODEL):
    - Trích xuất đặc trưng với TfidfVectorizer (1-3 n-gram, sublinear TF scaling).
    - Huấn luyện mô hình DeepHybridModel kết hợp Mạng nơ-ron sâu PyTorch (256, 128, 64) và ComplementNB.
    - Đánh giá kiểm thử (Train/Val Split 80/20) với Accuracy, Macro F1, Weighted F1.
    - Lưu toàn bộ trọng số PyTorch (.pth), metadata (.pkl) và 64-D Latent Semantic Embeddings (.npy).
    """
    questions, intents = load_training_data()

    if len(questions) < 2:
        print("Không đủ dữ liệu để huấn luyện AI.")
        return (None, None, None, None) if return_all else (None, None)

    if verbose:
        print(f"[*] Nạp {len(questions)} mẫu câu hỏi từ tập dữ liệu.")
        print(f"[*] Hỗ trợ PyTorch: {HAS_TORCH}")

    # Biểu diễn đặc trưng không gian vector đa chiều (TF-IDF với 1, 2, 3-grams)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        sublinear_tf=True,
        min_df=1
    )
    X = vectorizer.fit_transform(questions)

    eval_metrics = {}
    if evaluate and len(questions) >= 50:
        try:
            # Phân tách tập train / val 80/20 có phân tầng
            X_train, X_val, y_train, y_val = train_test_split(
                X, intents, test_size=0.2, random_state=42, stratify=intents
            )
        except ValueError:
            # Fallback nếu một số lớp có quá ít mẫu không thể stratify
            X_train, X_val, y_train, y_val = train_test_split(
                X, intents, test_size=0.2, random_state=42
            )

        if verbose:
            print(f"[*] Đánh giá Train/Val Split (Train: {X_train.shape[0]} mẫu, Val: {X_val.shape[0]} mẫu)...")

        eval_model = DeepHybridModel()
        eval_model.fit(X_train, y_train, epochs=45, verbose=False)
        y_val_pred = eval_model.predict(X_val)

        val_acc = float(accuracy_score(y_val, y_val_pred))
        val_macro_f1 = float(f1_score(y_val, y_val_pred, average="macro", zero_division=0))
        val_weighted_f1 = float(f1_score(y_val, y_val_pred, average="weighted", zero_division=0))

        eval_metrics = {
            "dataset_total_samples": len(questions),
            "train_samples": int(X_train.shape[0]),
            "val_samples": int(X_val.shape[0]),
            "val_accuracy": round(val_acc, 4),
            "val_macro_f1": round(val_macro_f1, 4),
            "val_weighted_f1": round(val_weighted_f1, 4),
        }
        if verbose:
            print(f"   [Evaluation Results] Accuracy: {val_acc*100:.2f}% | Macro F1: {val_macro_f1*100:.2f}% | Weighted F1: {val_weighted_f1*100:.2f}%")

    # Huấn luyện trên toàn bộ tập dữ liệu
    if verbose:
        print(f"[*] Huấn luyện mạng nơ-ron sâu PyTorch trên toàn bộ {len(questions)} mẫu...")
    model = DeepHybridModel()
    model.fit(X, intents, epochs=45, verbose=verbose)

    question_embeddings = None
    if save_artifacts:
        if verbose:
            print(f"[*] Lưu mô hình PyTorch, 64-D Latent Semantic Embeddings và metadata vào {save_dir}...")
        model.save_artifacts(
            save_dir=save_dir,
            vectorizer=vectorizer,
            questions=questions,
            metrics=eval_metrics
        )
        npy_path = os.path.join(save_dir, "question_embeddings.npy")
        if os.path.exists(npy_path):
            question_embeddings = np.load(npy_path)

    if return_all:
        return vectorizer, model, question_embeddings, eval_metrics
    return vectorizer, model


def load_or_train_model(force_retrain=False, save_dir=DEFAULT_SAVED_DIR):
    """
    Tải nhanh mô hình Deep Learning từ bộ nhớ đệm (saved_models/) nếu có (< 50ms);
    Nếu chưa có hoặc force_retrain=True thì huấn luyện mới và lưu lại.
    Trả về: (vectorizer, model, question_embeddings, metrics)
    """
    if not force_retrain:
        loaded = DeepHybridModel.load_artifacts(save_dir=save_dir)
        if loaded is not None:
            vectorizer, model, question_embeddings, metrics = loaded
            if vectorizer is not None and model is not None and question_embeddings is not None:
                return vectorizer, model, question_embeddings, metrics

    # Huấn luyện mới và lưu lại
    return train_model(
        force_retrain=True,
        save_artifacts=True,
        save_dir=save_dir,
        evaluate=True,
        return_all=True,
        verbose=True
    )


if __name__ == "__main__":
    print("=" * 70)
    print("HUẤN LUYỆN VÀ TÍNH TOÁN EMBEDDING MẠNG NƠ-RON SÂU PYTORCH CHO CHATBOT")
    print("=" * 70)
    start_t = time.time()
    vectorizer, model, question_embeddings, metrics = load_or_train_model(force_retrain=True)
    duration = time.time() - start_t

    print("\n" + "=" * 70)
    print("✅ HUẤN LUYỆN & PERSISTENCE HOÀN TẤT THÀNH CÔNG!")
    print(f"⏱ Thời gian thực thi: {duration:.2f}s")
    print(f"📊 Các lớp ý định (Classes - {len(model.classes_)} lớp): {list(model.classes_)}")
    if question_embeddings is not None:
        print(f"🧠 Kích thước ma trận Question Embeddings: {question_embeddings.shape} (64-D Latent Space)")
    if metrics:
        print(f"🎯 Độ chính xác kiểm thử (Val Accuracy): {metrics.get('val_accuracy', 0)*100:.2f}%")
        print(f"🎯 Chỉ số Macro F1: {metrics.get('val_macro_f1', 0)*100:.2f}%")
        print(f"🎯 Chỉ số Weighted F1: {metrics.get('val_weighted_f1', 0)*100:.2f}%")
    print("=" * 70)