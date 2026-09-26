"""
tools/generate_and_train.py
CÔNG CỤ TỰ ĐỘNG SINH CÂU HỎI, THU HOẠCH DỮ LIỆU & TÁI HUẤN LUYỆN CHATBOT
(AUTOMATED TRAVEL QA GENERATOR & DEEP LEARNING CONTINUAL RETRAINING TOOL)

Cách sử dụng:
  # Chạy sinh 30 câu hỏi và tự động tái huấn luyện:
  python tools/generate_and_train.py --num-samples 30

  # Chạy thử nghiệm không lưu DB (dry-run):
  python tools/generate_and_train.py --num-samples 15 --dry-run

  # Sinh riêng danh mục khách sạn và nhóm đoàn:
  python tools/generate_and_train.py --categories hotel,group --num-samples 20

  # Xuất kết quả ra file JSON tùy chọn:
  python tools/generate_and_train.py --export data/custom_generated_qa.json
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime

# Đảm bảo import được các module từ thư mục gốc
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ai.qa_generator import TravelQuestionGenerator, DataHarvestingTrainingPipeline
from ai.chatbot import Chatbot


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bộ công cụ tự động sinh câu hỏi, thu hoạch phản hồi Chatbot và tái huấn luyện mô hình Deep Learning."
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        default=30,
        help="Số lượng câu hỏi cần sinh tự động (Mặc định: 30)."
    )
    parser.add_argument(
        "--categories",
        "-c",
        type=str,
        default=None,
        help="Danh sách các danh mục ngăn cách bằng dấu phẩy: hotel,group,budget,schedule,weather,policy,overseas."
    )
    parser.add_argument(
        "--no-retrain",
        action="store_true",
        help="Chỉ sinh và lưu dữ liệu, không thực hiện tái huấn luyện mô hình."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chạy mô phỏng kiểm thử: sinh và hỏi nhưng không lưu DB và không retrain."
    )
    parser.add_argument(
        "--export",
        "-e",
        type=str,
        default="data/generated_training_qa.json",
        help="Đường dẫn file JSON xuất tập dữ liệu đã thu hoạch (Mặc định: data/generated_training_qa.json)."
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Giảm thiểu hiển thị chi tiết ra màn hình."
    )
    return parser.parse_args()


def print_banner():
    banner = """
================================================================================
🤖 TOURAI - AUTOMATED QA GENERATOR & DEEP LEARNING RETRAINING PIPELINE
   Hệ thống Tự động Sinh câu hỏi, Thu hoạch Dữ liệu và Tái huấn luyện Mạng Nơ-ron Sâu
================================================================================
"""
    print(banner)


def main():
    args = parse_args()
    print_banner()

    start_total_time = time.time()
    verbose = not args.quiet

    categories = None
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",") if c.strip()]
        print(f"[*] Danh mục được chọn: {', '.join(categories)}")

    print(f"[*] Số lượng câu hỏi yêu cầu sinh: {args.num_samples}")
    print(f"[*] Chế độ Dry-Run: {args.dry_run}")
    print(f"[*] Tái huấn luyện mô hình: {not args.no_retrain and not args.dry_run}")
    print(f"[*] Đường dẫn xuất dữ liệu: {args.export}")
    print("-" * 80)

    # 1. Khởi tạo Chatbot & Pipeline
    print("[1/4] Khởi tạo Chatbot AI & Bộ máy Thu hoạch...")
    bot = Chatbot()
    pipeline = DataHarvestingTrainingPipeline(chatbot_instance=bot)

    # 2. Sinh câu hỏi & Thu hoạch phản hồi
    print(f"\n[2/4] Sinh ngẫu nhiên và gửi {args.num_samples} câu hỏi qua Chatbot...")
    should_retrain = not args.no_retrain and not args.dry_run
    res = pipeline.run_full_pipeline(
        num_samples=args.num_samples,
        categories=categories,
        dry_run=args.dry_run,
        auto_retrain=should_retrain,
        export_path=args.export,
        verbose=verbose
    )

    harvested_data = res["harvested_data"]
    valid_count = res["valid_count"]
    invalid_count = res["invalid_count"]

    # Thống kê Intent và Phân loại
    intent_counts = {}
    cat_counts = {}
    latencies = []

    for h in harvested_data:
        it = h.get("intent", "khac")
        intent_counts[it] = intent_counts.get(it, 0) + 1
        cat = h.get("category", "khac")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        latencies.append(h.get("latency_ms", 0))

    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0

    print("\n" + "=" * 80)
    print("📊 KẾT QUẢ THU HOẠCH DỮ LIỆU (HARVESTING SUMMARY)")
    print("=" * 80)
    print(f"  • Tổng số câu hỏi đã sinh: {res['total_generated']}")
    print(f"  • Phản hồi đạt chuẩn chất lượng: {valid_count} ({valid_count/res['total_generated']*100:.1f}%)")
    print(f"  • Phản hồi không đạt/lỗi: {invalid_count}")
    print(f"  • Độ trễ trung bình: {avg_latency} ms/câu hỏi")
    print("\n📌 Phân bố theo Chuyên mục (Categories):")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    - {cat:<15}: {cnt:>3} mẫu ({cnt/len(harvested_data)*100:4.1f}%)")

    print("\n📌 Phân bố theo Ý định (Intents):")
    for it, cnt in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    - {it:<20}: {cnt:>3} mẫu ({cnt/len(harvested_data)*100:4.1f}%)")

    # 3. Nạp dữ liệu vào CSDL nếu không phải dry-run
    ingest_info = res["ingest_result"]
    if not args.dry_run:
        print("\n" + "=" * 80)
        print("💾 KẾT QUẢ ĐỒNG BỘ CƠ SỞ TRI THỨC (DATABASE & STORAGE)")
        print("=" * 80)
        print(f"  • Số bản ghi mới đã nạp vào MySQL (qa_data) & sample_qa.json: {ingest_info.get('added_count', 0)}")
        print(f"  • Số bản ghi trùng lặp được lọc bỏ: {ingest_info.get('duplicates_skipped', 0)}")

        # 4. Hiển thị thông số Tái huấn luyện mô hình PyTorch Deep Learning
        if should_retrain and ingest_info.get("added_count", 0) > 0:
            retrain_res = res.get("retrain_result", {})
            metrics = retrain_res.get("metrics", {})

            print("\n" + "=" * 80)
            print("🧠 KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH HỌC SÂU (EVALUATION METRICS)")
            print("=" * 80)
            print(f"  • Thời gian huấn luyện: {retrain_res.get('duration_seconds', 0)} giây")
            print(f"  • Tổng số mẫu trong tập dữ liệu: {metrics.get('dataset_total_samples', 'N/A')}")
            print(f"  • Tập Train: {metrics.get('train_samples', 'N/A')} mẫu | Tập Validation: {metrics.get('val_samples', 'N/A')} mẫu")
            print(f"  • Validation Accuracy: {metrics.get('val_accuracy', 0)*100:.2f}%")
            print(f"  • Validation Macro F1: {metrics.get('val_macro_f1', 0)*100:.2f}%")
            print(f"  • Validation Weighted F1: {metrics.get('val_weighted_f1', 0)*100:.2f}%")
            print("  • Vector nhúng ngữ nghĩa sâu 64-D (64-D Latent Semantic Space): Đã cập nhật (.npy)")
            print("  • Trọng số PyTorch (deep_intent_model.pth): Đã lưu thành công")
            print("  • Hot-reload Chatbot: Đã kích hoạt tri thức mới trong RAM")
        elif ingest_info.get("added_count", 0) == 0:
            print("\n[*] Không có câu hỏi mới nào chưa từng có trong hệ thống; giữ nguyên mô hình hiện tại.")
    else:
        print("\n[!] Bỏ qua giai đoạn ghi CSDL và Tái huấn luyện mô hình (Chế độ Dry-Run hoặc No-Retrain).")

    total_duration = round(time.time() - start_total_time, 2)
    print("\n" + "=" * 80)
    print(f"🎉 HOÀN TẤT TOÀN BỘ QUY TRÌNH TRONG {total_duration} GIÂY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
