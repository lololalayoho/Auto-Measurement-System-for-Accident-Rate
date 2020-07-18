# Auto-Measurement-System-for-Accident-Rate

## CCTV 영상을 이용하여 자동차 사고 과실 비율을 자동으로 측정하는 시스템입니다.

<img src = "/image/Resultvideo.gif">

## 개발 환경
* Ubuntu 18.04
* Python 3.6.9
* Tensorflow 1.14.0
* Keras 2.3.1
* CUDA 10.2
* CUDNN 7.6
* PYQT5

## 실행 방법
<pre>
git clone https://github.com/lololalayoho/Auto-Measurement-System-for-Accident-Rate.git
cd Auto-Measurement-System-for-Accident-Rate
wget https://pjreddie.com/media/files/yolov3.weights
python covert.py yolov3.cfg yolov3.weigths model_data/yolo.h5
python gui.py
</pre>

* mask r cnn.h5 파일이 없을 시 gui.py 실행하면 자동으로 다운 (다운 받는 시간이 소요됨)

## 참고주소

#### YOLO + Deepsort : https://github.com/Qidian213/deep_sort_yolov3
#### Mask R CNN : https://github.com/matterport/Mask_RCNN
#### 3D CNN : https://github.com/lianggyu/C3D-Action-Recognition

## Paper & 설명 URL

[교통사고 과실비율 자동측정 시스템_최진우_이유노 논문](https://github.com/lololalayoho/Auto-Measurement-System-for-Accident-Rate/blob/master/%EA%B5%90%ED%86%B5%EC%82%AC%EA%B3%A0%20%EA%B3%BC%EC%8B%A4%EB%B9%84%EC%9C%A8%20%EC%9E%90%EB%8F%99%EC%B8%A1%EC%A0%95%20%EC%8B%9C%EC%8A%A4%ED%85%9C_%EC%B5%9C%EC%A7%84%EC%9A%B0_%EC%9D%B4%EC%9C%A0%EB%85%B8.pdf)

#### 프로젝트 설명 url : https://www.youtube.com/watch?v=dVQ79Bnkgn4&t=569s
