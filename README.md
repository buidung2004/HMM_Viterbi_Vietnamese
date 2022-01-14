# CS221 - GÁN NHÃN TỪ LOẠI TIẾNG VIỆT

<p align="center">
  <img src="https://user-images.githubusercontent.com/56221762/111880949-da1dd580-89e0-11eb-876c-a68752260d3b.png">
</p>

# Giới thiệu

This repository stores our Capstone Project of CS221 - Natural Language Processing.

## Thành viên

|Order|    Member         |  ID        | Role 
|:---:| :-----------:     | :--:       | :--: 
|1    |    [Dung B.Ngoc](https://github.com/buidung2004/)     |  19521385  | Leader
|2    |    [Minh N.Dang](https://github.com/ELO102)    |  19520164  | Member

# Mô tả bài toán 
Chúng em xây dựng mô hình gán nhãn tự động cho tiếng việt theo nguyên tắc đã quy ước sau: 
https://github.com/vncorenlp/VnCoreNLP/blob/master/VLSP2013_POS_tagset.pdf
# Các bước thực hiện
- Thực hiện tách từ bằng thuật toán Maximum Matching và so sánh với phương pháp tách từ tự động VNCoreNLP
- Xây dựng mô hình gán nhãn Hidden Markov và sử dụng thuật toán Viterbi để decoding
- Đánh giá kết quả trên tập dữ liệu của nhóm, so sánh với cách gán nhãn bằng thư viện VnCoreNLP
- Nhận xét
Clone github:
```
git clone https://github.com/buidung2004/POS-Tagging-Vietnamese
cd POS-Tagging-Vietnamese/WordSegmentation
pip install -r requirements.txt

```
# Tách từ bằng Maximum Matching
Xây dựng bộ vocabs.txt chứa các từ có trong dữ liệu huấn luyện, có thể nhiều hơn để đa dạng
```
cd POS-Tagging-Vietnamese/WordSegmentation
python main.py --path_data "đường dẫn đến các raw_sentence" --max_len "độ dài lớn nhất có thể tách"
```
# Training
Training:
```
cd POS-Tagging-Vietnamese
python train.py --path_train "đường dẫn đến dữ liệu huấn luyện đã gán nhãn" --path_dict "đường dẫn đến từ điển đã được xây dựng"
```
các tham số và ma trận của mô hình được lưu vào ```dir_save```
# Evaluation
Sử dụng thuật toán viterbi để gán nhãn và đánh giá so với nhãn đúng:
```
cd POS-Tagging-Vietnamese
python eval_with_viterbi.py 
		--path_test "đường dẫn dữ liệu test"
		--path_path_test_no_tag "đường dẫn dữ liệu test xóa nhãn đi"
		--path_A "đường dẫn lưu ma trận chuyển trạng thái (trasition matrix)"
		--path_B "đường dẫn lưu ma trận thể hiện (emisson matrix)"
		--path_tag "đường dẫn lưu các nhãn đã huấn luyện"
```
# Testing 
Thử gán nhãn với một câu bất kỳ:
```
cd POS-Tagging-Vietnamese
python test.py --sentence "Đặt một câu ví dụ bất kỳ với dấu chấm cuối câu cách ra"
```
# Running webapp demo
```
cd POS-Tagging-Vietnamese
python app.py
```
![Screen of Flask App](app.jpg)

# Reference
- https://github.com/alejandropuerto/viterbi-HMM-POS-tagging
- https://github.com/18520339/vietnamese-pos-tagging




