import pickle
from flask import Flask, render_template, request
import os
from random import random
from my_yolov6 import my_yolov6
import cv2
from paddleocr import PaddleOCR
import numpy as np
import csv
yolov6_model = my_yolov6("weights/best_ckpt.pt","cpu","data/mydataset.yaml", 640, True)
names = ['dog','person', 'cat','tv','car','meatballs','marinara sauce','tomato soup','chicken noodle soup','french onion soup','chicken breast','ribs','pulled pork','hamburger','cavity','company','address','total','date']
# Khởi tạo Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static"
ocr = PaddleOCR()
text_info = []
# Hàm xử lý request
@app.route("/", methods=['GET', 'POST'])
def home_page():
    # Nếu là POST (gửi file)
    if request.method == "POST":
         try:
            # Lấy file gửi lên
            image = request.files['file']
            if image:
                # Lưu file
                path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
                print("Save = ", path_to_save)
                image.save(path_to_save)

                # Convert image to dest size tensor
                frame = cv2.imread(path_to_save)
                frame, det ,ndet = yolov6_model.infer(frame, conf_thres=0.4, iou_thres=0.45)

                if ndet !=0:
                    imageRecognize = cv2.imread(path_to_save)
                    with open('Output/output.csv', mode='a', newline='', encoding='utf-8') as file:
                        writer = csv.writer(file)
                        for row in det.tolist():
                            x1, y1, x2, y2 = map(int, row[:4])
                            label = (int)(row[-1])
                            if isinstance(imageRecognize, np.ndarray):
                                text_region = imageRecognize[y1:y2, x1:x2].copy()
                                result = ocr.ocr(text_region)
                                for region in result:
                                    combined_text = ""
                                    for polygon, (text, confidence) in region:
                                        combined_text += text + " "
                                    writer.writerow([image.filename,names[label], combined_text])

                            else:
                                print("Không thể cắt vùng văn bản từ ảnh.")
                    cv2.imwrite(path_to_save, frame)

                    # Trả về kết quả
                    return render_template("index.html", user_image = image.filename , rand = str(random()),
                                           msg="Tải file lên thành công", ndet = ndet)
                else:
                    return render_template('index.html', msg='Không nhận diện được vật thể')
            else:
                # Nếu không có file thì yêu cầu tải file
                return render_template('index.html', msg='Hãy chọn file để tải lên')

         except Exception as ex:
            # Nếu lỗi thì thông báo
            print(ex)
            return render_template('index.html', msg='Không nhận diện được vật thể')

    else:
        # Nếu là GET thì hiển thị giao diện upload
        return render_template('index.html')

def extract_text_from_ocr_result(ocr_result):
    texts = []
    for sublist in ocr_result:
        for item in sublist:
            text = item[1][0]
            texts.append(text)
    return texts
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
