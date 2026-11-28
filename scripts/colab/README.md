# Chạy pipeline trên Colab từ terminal (google-colab-cli)

[google-colab-cli](https://github.com/googlecolab/google-colab-cli) cấp VM Colab (GPU T4), chạy file `.py` / `.ipynb` trên VM đó và tải file về, **không cần mở trình duyệt để chạy cell**. CLI chỉ hỗ trợ Linux/macOS → trên Windows dùng **WSL Ubuntu**.

## Cài đặt (một lần)

```bash
# trong WSL
cd /mnt/d/AiThucChien/other/hackathon-l2-team56
bash scripts/colab/setup_wsl.sh
source ~/.bashrc
```

Token đăng nhập Google được lưu bằng `keyrings.alt` (file chỉ user WSL đọc được), vì WSL không có keyring của hệ điều hành.

## Các bước (R0 → lượt nộp #1)

| Lệnh | Làm gì | Thời gian |
|---|---|---|
| `scripts/colab/pipeline.sh new` | Tạo VM T4 tên `team56`. Lần đầu: mở URL in ra, đăng nhập, dán mã xác thực | 1 phút |
| `scripts/colab/pipeline.sh mount` | Mount Google Drive vào VM: mở URL in ra và bấm đồng ý (đúng tài khoản `colab whoami`) | 1 phút |
| `scripts/colab/pipeline.sh prepare` | Clone repo, cài thư viện, giải nén `roboflow_790.zip`, `kaggle_3000.zip`, `golden_val_150.zip` từ Drive; tạo `kaggle_day` (bỏ ảnh đêm) | 3–5 phút |
| `scripts/colab/pipeline.sh autolabel` | Gán nháp Roboflow 790 + 1.000 ảnh Kaggle ban ngày → `Drive/team56/drafts/`, kèm bảng lớp + validator | ~70 phút |
| `scripts/colab/pipeline.sh build r00` | Build `du_lieu_r00.zip` → Drive; tải `manifests/du_lieu_r00.csv/.json` về repo | ~10 phút |
| `scripts/colab/pipeline.sh train r00 40` | Chạy notebook BTC (`notebooks/runs/train_r00.ipynb`, chỉ ô tham số khác bản gốc) → train, xuất ONNX, self-check, `submission.zip` | ~2 giờ |
| `scripts/colab/pipeline.sh fetch r00` | Tải `submission.zip` + `run_report.json` về `experiments/round_00/` | 1 phút |
| `scripts/colab/pipeline.sh stop` | Trả VM | — |

`scripts/colab/pipeline.sh r0` chạy tất cả theo thứ tự (bước `mount` vẫn cần bấm đồng ý trên trình duyệt).

Mỗi bước gọi `colab exec --timeout 0` nên job chạy lâu không bị cắt; log hiện trực tiếp trong terminal. Nếu mạng rớt giữa chừng: VM vẫn chạy tiếp, chạy lại đúng bước đó (autolabel và build đều chạy tiếp được, `best.pt` được chép sang `Drive/team56/outputs/<round>`).

Biến môi trường: `COLAB_SESSION` (mặc định `team56`), `COLAB_GPU` (`T4`), `REPO_REF` (`main`), `KAGGLE_SAMPLE` (`1000`), `SOURCES` (`roboflow_kien,kaggle_day`), `USE_SAM` (`1`), `EPOCHS` (cho bước `r0`, `40`).

## Sau khi `fetch`

1. Nộp `experiments/round_00/submission.zip` trên website (Nộp checkpoint).
2. Commit `experiments/round_00/run_report.json`, `manifests/du_lieu_r00.*`, `notebooks/runs/train_r00.ipynb` (zip và `*_output.ipynb` không commit).
