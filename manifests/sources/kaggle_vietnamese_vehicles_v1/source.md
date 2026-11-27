# Vietnamese Vehicles Dataset — hồ sơ nguồn Kaggle

- Nguồn: https://www.kaggle.com/datasets/duongtran1909/vietnamese-vehicles-dataset
- Người cung cấp: Duong Thanh Tran (`duongtran1909`).
- Phiên bản: 1; metadata cập nhật 2022-03-22.
- License công bố: **CC0: Public Domain**.
- Điều kiện CC0: https://creativecommons.org/publicdomain/zero/1.0/
- Metadata chính thức: https://www.kaggle.com/api/v1/datasets/view/duongtran1909/vietnamese-vehicles-dataset
- Bằng chứng: `kaggle-metadata.json`, lưu phản hồi API chính thức ngày 2026-11-27.
- Ngày tải ban đầu: chưa ghi nhận.
- Thư mục nguồn local: `E:\AI-Vin-Lab\archive` (được người dùng xác định là bộ Kaggle này).
- Ảnh trong repo: `data/raw/kaggle_vietnamese_vehicles_v1/` (49.270 file ảnh, chưa dedup).
- Số ảnh dùng train/val và số instance: chưa chốt.

## Sử dụng và ghi nguồn

Đội chỉ lấy ảnh, không sử dụng annotation gốc. Kế hoạch là tạo nhãn mới theo guideline 11 lớp BTC. Ảnh được copy giữ cấu trúc thư mục; chưa xử lý lại pixel trong bước copy.

CC0 cho phép sao chép, sửa đổi và phân phối, kể cả thương mại, không yêu cầu ghi công theo công cụ CC0. Đội vẫn dẫn tên người cung cấp, link nguồn, phiên bản và license trong báo cáo để truy xuất được dữ liệu.

Ghi nguồn đề xuất: Vietnamese Vehicles Dataset — Duong Thanh Tran, Kaggle, v1, CC0: Public Domain. Nguồn và CC0 theo các link trên. Đội chỉ sử dụng ảnh và tạo lại nhãn theo guideline 11 lớp.

## Giới hạn xác minh

API xác nhận trường `licenseName` là `CC0: Public Domain`; trường `description` đang trống. Đây là license do người đăng khai báo; chưa xác minh độc lập nguồn gốc/quyền sở hữu từng ảnh. Hồ sơ này không xác nhận bộ ảnh có nguồn gốc độc lập với các bộ khác; cần dedup trước khi chia train/val.

Bảng dùng cho báo cáo nằm ở repo `hackathon-l2-docs`, file `plan/datasets.md`. Ảnh nằm trong `data/` và không commit; hồ sơ nguồn này được lưu Git.
