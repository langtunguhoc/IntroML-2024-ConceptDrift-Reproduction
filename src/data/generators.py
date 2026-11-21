from river.datasets import synth


def get_concept_drift_stream(drift_type="sudden", n_classes=2, seed=42):
    """Tạo luồng dữ liệu tổng hợp từ RandomRBF Generator có chèn vị trí Concept Drift tại điểm 50,000."""
    stream_base = synth.RandomRBF(
        seed_model=seed, n_classes=n_classes, n_features=5
    )
    stream_drift = synth.RandomRBF(
        seed_model=seed + 100, n_classes=n_classes, n_features=5
    )

    if drift_type == "sudden":
        # Sudden Drift: Thay đổi đột ngột ngay mẫu thứ 50,000
        stream = synth.ConceptDriftStream(
            stream=stream_base,
            drift_stream=stream_drift,
            position=50000,
            width=1000,
        )
    else:
        # Gradual Drift: Chuyển tiếp từ từ trong cửa sổ 2,000 mẫu
        stream = synth.ConceptDriftStream(
            stream=stream_base,
            drift_stream=stream_drift,
            position=50000,
            width=2000,
        )
    return stream
