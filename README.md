# Team 56 — AIInAction RoboTaxi Data-Centric Competition

Instance segmentation 11 lớp cho ảnh giao thông Việt Nam. Model cố định là **YOLO26n-seg** (`imgsz=640`), nên đội chỉ cải thiện **dữ liệu**.
Điểm = 100 × mask mAP@[.5:.95]. **TEAM_ID / seed = 61.**

- Đề bài, guideline gán nhãn, kế hoạch, nhật ký thí nghiệm và slide nằm ở repo `hackathon-l2-docs`.
- Repo này chứa phần làm bài: code chuẩn bị dữ liệu, notebook train/export, config, manifest dữ liệu và `run_report.json`.

## Cấu trúc

```
configs/        classes.yaml (11 lớp cố định), sources.yaml (public dataset + ánh xạ lớp), train/round_XX.yaml
src/robotaxi/   thư viện dùng chung: lớp, đọc/ghi nhãn YOLO-seg, xử lý mask, gộp rider
scripts/data/   chuyển public dataset → YOLO-seg, SAM box→polygon, build dataset zip, dedup
scripts/label/  gán nháp tự động, active learning, CVAT import/export
scripts/qa/     validate nhãn, lưới ảnh duyệt, thống kê lớp
notebooks/      train_and_export.ipynb (notebook BTC, TEAM_ID=61), autolabel_kaggle.ipynb
manifests/      danh sách ảnh + nguồn của từng phiên bản dữ liệu
experiments/    round_XX/: run_report.json, kết quả val, AP từng lớp
data/           (không commit) raw/, interim/, processed/
```

Ảnh, zip dữ liệu, `.pt` và `.onnx` **không** được commit. Chúng nằm trên Kaggle Dataset hoặc Google Drive; sha256 được ghi trong manifest và trong `run_report.json`.

## Cài đặt tool local

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
```

## Chạy trên Colab từ terminal

Xem [scripts/colab/README.md](scripts/colab/README.md): `google-colab-cli` (qua WSL) chạy gán nháp, build zip, train và tải `submission.zip` bằng `scripts/colab/pipeline.sh`.

## Tái lập kết quả

1. Dựng zip dữ liệu từ manifest (sẽ bổ sung ở `scripts/data/build_dataset.py`).
2. Mở `notebooks/train_and_export.ipynb` trên Kaggle/Colab (GPU), trỏ `DATASET_PATH` tới zip, giữ `TEAM_ID = 61`, rồi chạy hết các ô.
3. So sánh `run_report.json` sinh ra với file tương ứng trong `experiments/round_XX/`.
