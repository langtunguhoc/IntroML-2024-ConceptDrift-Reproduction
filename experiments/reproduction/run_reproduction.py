import os
import sys

# Thêm thư mục gốc vào PYTHONPATH để nhận diện package 'src'
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)

from river import drift, metrics, tree
from src.data.generators import get_concept_drift_stream
import wandb


def run_experiment(detector_name="ADWIN", drift_type="sudden"):
    # 1. Khởi tạo Weights & Biases Run
    run = wandb.init(
        project="IntroML-ConceptDrift-Reproduction",
        name=f"{detector_name}_{drift_type}",
        config={
            "detector": detector_name,
            "drift_type": drift_type,
            "classifier": "HoeffdingTree",
            "max_instances": 100000,
        },
    )

    # 2. Khởi tạo Stream, Classifier và Detector
    stream = get_concept_drift_stream(drift_type=drift_type)
    model = tree.HoeffdingTreeClassifier()

    if detector_name == "ADWIN":
        detector = drift.ADWIN()
    elif detector_name == "PageHinkley":
        detector = drift.PageHinkley()
    else:
        detector = drift.DDM()

    metric = metrics.Accuracy()
    drift_count = 0

    print(f"--- Bắt đầu chạy thực nghiệm: {detector_name} ({drift_type}) ---")

    # 3. Vòng lặp Prequential Evaluation (Test-then-Train)
    for i, (x, y) in enumerate(stream.take(100000)):
        # Dự đoán trước (Test)
        y_pred = model.predict_one(x)

        if y_pred is not None:
            metric.update(y, y_pred)
            error = 0 if y_pred == y else 1
            detector.update(error)

            # Cảnh báo phát hiện Drift
            if detector.drift_detected:
                drift_count += 1
                wandb.log({"drift_detected_event": 1, "instance": i})
                print(
                    f"-> [Báo động Drift #{drift_count}] Phát hiện tại mẫu thứ: {i}"
                )

        # Cập nhật học online (Train)
        model.learn_one(x, y)

        # Log chỉ số lên W&B định kỳ mỗi 1.000 mẫu
        if i % 1000 == 0 and i != 0:
            wandb.log(
                {
                    "instance": i,
                    "accuracy": metric.get() * 100,
                    "total_drifts": drift_count,
                }
            )

    print(
        f"Hoàn tất! Độ chính xác cuối cùng: {metric.get() * 100:.2f}%\n"
    )
    wandb.finish()


if __name__ == "__main__":
    # Chạy thử 2 bộ phát hiện Drift tiêu biểu
    run_experiment(detector_name="ADWIN", drift_type="sudden")
    run_experiment(detector_name="PageHinkley", drift_type="sudden")
    run_experiment(detector_name="ADWIN", drift_type="gradual")
    run_experiment(detector_name="PageHinkley", drift_type="gradual")
