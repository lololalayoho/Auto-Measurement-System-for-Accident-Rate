from __future__ import division, print_function, absolute_import
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
import os
from timeit import time
import warnings
import math
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
from models import c3d_model
from keras.optimizers import SGD
import configparser
from divide_score import divide
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')
video_classification = []
for i in range(10):
    video_classification.append(0)

form_class = uic.loadUiType("accident_gui.ui")[0]
class MyWindow(QDialog, form_class):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setFixedSize(1200, 700)
        self.setupUI()

    def show_videoo(self,aframe):
       # cv2.imshow('',aframe)
        aframe = cv2.cvtColor(aframe,cv2.COLOR_BGR2RGB)
        h, w, c= aframe.shape
        qImg = QImage(aframe.data,w,h,QImage.Format_RGB888)
        pixmap = QPixmap(qImg)
        QApplication.processEvents()
        self.show_video.setPixmap(pixmap)
        self.show_video.update()        
        #self.show()

    def pushButtonClicked(self):
        fname = QFileDialog.getOpenFileName(self)
        self.addressplain.setText(fname[0])

    def check_address(self):
        self.faddress=self.addressplain.toPlainText()
        self.progress.setValue(50)
        self.main(YOLO(),self.faddress)

    def setupUI(self):
        self.setupUi(self)
        self.progress = QProgressBar(self)
        self.progress.setGeometry(50,100,300,25)
        self.progress.setMaximum(100)
        self.file.clicked.connect(self.pushButtonClicked)
        self.check.clicked.connect(self.check_address)

    def main(self,yolo,video_stream):
        loading = cv2.imread("loading.jpeg")
        loading = cv2.resize(loading,(400,250))
        loading = cv2.cvtColor(loading, cv2.COLOR_BGR2GRAY)
        crashh, crashw = loading.shape
        crashqImg = QImage(loading.data, crashw, crashh, QImage.Format_Indexed8)
        crashpixmap = QPixmap(crashqImg)
        self.mask.setPixmap(crashpixmap)
        self.mask.update()

        maskCrash = False
        crashcar = [[]for i in range(2)]
        crashcarnum = [0,0]
        CRASH = True
        self.progress.setValue(55)
        root_dir=os.path.abspath(os.path.dirname(__file__))
        configpath = os.path.join(root_dir, "config.txt")
        self.progress.setValue(60)
        self.progress.setValue(70)
        config2 = configparser.ConfigParser()
        config2.read(configpath)
        classInd_path = config2.get("C3D", "classInd_path")
        weights_path = config2.get("C3D", "weights_path")
        lr = config2.get("C3D", "lr")
        momentum = config2.get("C3D", "momentum")
        self.progress.setValue(80)
        video_image = config2.get("choose", "video_image")
        with open(classInd_path, 'r') as f:
            class_names = f.readlines()
            f.close()

        c3model = c3d_model()
        sgd = SGD(lr=float(lr), momentum=float(momentum), nesterov=True)
        c3model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        c3model.summary()
        self.progress.setValue(90)
        c3model.load_weights('results/weights_c3d.h5', by_name=True)


        first = True
        def multi_detecion(clip, frame):
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs[..., 0] -= 99.9
            inputs[..., 1] -= 92.1
            inputs[..., 2] -= 82.6
            inputs[..., 0] /= 65.8
            inputs[..., 1] /= 62.3
            inputs[..., 2] /= 60.3
            inputs = inputs[:, :, 0:112, 0:112, :]
            inputs = np.transpose(inputs, (0, 2, 3, 1, 4))
            pred = c3model.predict(inputs)
            label = np.argmax(pred[0])
            cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (40, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0, 0, 255), 3)
            cv2.putText(frame, "prob: %.4f" % pred[0][label], (40, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0, 0, 255), 3)
            index = int(class_names[label].split(' ')[0])
            if pred[0][label]>= 0.5:
                video_classification[index] = video_classification[index] + pred[0][label]
            clip.pop(0)
            return (frame)

        def crash_detect(fframe):
            from visualize_cv2 import modelhello,display_instances, class_names
            # add mask to frame
            results = modelhello.detect([fframe], verbose=0)
            if len(results) == 1 and first == True:
                return 0
            r = results[0]
            carframe = display_instances(
                fframe, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
            )
            image_gray = cv2.cvtColor(carframe,cv2.COLOR_BGR2GRAY)
            bb,contours = cv2.findContours(image_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            ret, thr2 = cv2.threshold(image_gray,255,255,cv2.THRESH_BINARY)

            
            for cnt in bb:
                ellipse = cv2.fitEllipse(cnt)
                cv2.ellipse(image_gray,ellipse,(255,255,255),-1)
            nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_gray)
            image_gray = cv2.resize(image_gray,(400,250))
            
            h, w = image_gray.shape
            qImg = QImage(image_gray.data, w, h, QImage.Format_Indexed8)
            pixmap = QPixmap(qImg)
            QApplication.processEvents()
            self.mask.setPixmap(pixmap)
            self.show()
            if nlabels == 2:
                return True
            else:
                return False


        # Definition of the parameters
        max_cosine_distance = 0.3
        nn_budget = None
        nms_max_overlap = 1.0

       # deep_sort 
        model_filename = 'model_data/mars-small128.pb'
        encoder = gdet.create_box_encoder(model_filename,batch_size=1)

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        tracker = Tracker(metric)

        if video_image=='video':
            video = []
            video_capture = cv2.VideoCapture(video_stream)
            dx = int(0)
            dy = int(0)
            car=[[] for i in range(200)]
            crash = [False for i in range(200)]
            w = int(video_capture.get(3))
            h = int(video_capture.get(4))
        
            fps = 0.0
        
            exx=[0 for i in range(200)]
            exy=[0 for i in range(200)]
            exd=[[0,0,0] for i in range(200)]
            cnt=int(0)
            self.progress.setValue(100)
            while CRASH:
                try:

                    
                    cararea=[(0,0,0,0) for i in range(200)]
                    #curframe = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
                    ret, frame = video_capture.read()  # frame shape 640*480*3
                    retcopy, framecopy = video_capture.read()
                    if ret != True:
                        break
                    tmp = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    video.append(cv2.resize(tmp,(112,112)))
                    if len(video)==16:
                        frame = multi_detecion(video,frame)
                    t1 = time.time()
            
                    image = Image.fromarray(frame[...,::-1]) #bgr to rgb
           
                    boxs = yolo.detect_image(image)
                    features = encoder(frame,boxs)
        
                    detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
                    if(first):
                        crash_detect(frame[1:10,1:10])
                        first = False


                    # Run non-maxima suppression.
                    boxes = np.array([d.tlwh for d in detections])
                    scores = np.array([d.confidence for d in detections])
                    indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
                    detections = [detections[i] for i in indices]
            
                    # Call the tracker
                    tracker.predict()
                    tracker.update(detections)
            
                    for track in tracker.tracks:
                        if cnt < track.track_id:
                            cnt = track.track_id
                        if not track.is_confirmed() or track.time_since_update > 1:
                            continue 
                        bbox = track.to_tlbr()
                        x1 = int(bbox[0])
                        x2 = int(bbox[2])
                        y1 = int(bbox[1])
                        y2 = int(bbox[3])
            
                        curx = (x1 + x2) / 2
                        cury = (y1 + y2) / 2
            
                        if exx[track.track_id] == 0:
                            exx[track.track_id] = curx
                            exy[track.track_id] = cury
            
                        dx = abs(curx-exx[track.track_id])
                        dy = abs(cury-exy[track.track_id])
                        exx[track.track_id] = curx
                        exy[track.track_id] = cury
                        if not crash[track.track_id]:
                            car[track.track_id].append((round(curx/(w/320),1),round(cury/(h/240),1),round(dx/(w/320),1),round(dy/(h/240),1)))
                        cararea[track.track_id]=(x1,y1,x2,y2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2),(255,0,35), 2)
                    resizeframe = cv2.resize(frame,(640,480)) 
                    self.show_videoo(resizeframe)
            
                    for i in range(cnt):
                        if not crash[i] and cararea[i][0] is not 0:
                            for j in range(cnt+1):
                                if i<j and not crash[j] and cararea[j][0] is not 0:
                                    if (cararea[i][0]<cararea[j][0] and cararea[i][2]>cararea[j][0] and cararea[i][1]<cararea[j][1] and cararea[i][3]>cararea[j][1]) or (cararea[i][0]<cararea[j][2] and cararea[i][2]>cararea[j][2] and cararea[i][1]<cararea[j][3] and cararea[i][3]>cararea[j][3]) or (cararea[i][0]<cararea[j][0] and cararea[i][2]>cararea[j][0] and cararea[i][1]<cararea[j][3] and cararea[i][3]>cararea[j][3]) or (cararea[i][0]<cararea[j][2] and cararea[i][2]>cararea[j][2] and cararea[i][1]<cararea[j][1] and cararea[i][3]>cararea[j][1]) or (cararea[j][0]<cararea[i][0] and cararea[j][2]>cararea[i][0] and cararea[j][1]<cararea[i][1] and cararea[j][3]>cararea[i][1]) or (cararea[j][0]<cararea[i][2] and cararea[j][2]>cararea[i][2] and cararea[j][1]<cararea[i][3] and cararea[j][3]>cararea[i][3]) or (cararea[j][0]<cararea[i][0] and cararea[j][2]>cararea[i][0] and cararea[j][1]<cararea[i][3] and cararea[j][3]>cararea[i][3]) or (cararea[j][0]<cararea[i][2] and cararea[j][2]>cararea[i][2] and cararea[j][1]<cararea[i][1] and cararea[j][3]>cararea[i][1]):
                                        minx = min(cararea[i][0], cararea[j][0])-25
                                        maxx = max(cararea[i][2], cararea[j][2])+25
                                        miny = min(cararea[i][1], cararea[j][1])-25
                                        maxy = max(cararea[i][3], cararea[j][3])+25
                                        maskCrash = crash_detect(framecopy[miny:maxy,minx:maxx])
                                        if maskCrash:
                                            crash[i] = crash[j] = True;
                                            CRASH = False
                                            cv2.imwrite("car%s.jpg"%(i),frame[cararea[i][1]:cararea[i][3],cararea[i][0]:cararea[i][2]])
                                            cv2.imwrite("car%s.jpg"%(j),frame[cararea[j][1]:cararea[j][3],cararea[j][0]:cararea[j][2]])
            
                    # Press Q to stop!
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception as e:
                    print(str(e))
        
            carnum = int(0)
            for i in range(cnt+1):
                if i is not 0 and crash[i]:
                     crashcar[carnum] = car[i]
                     crashcarnum[carnum] = i
                     carnum = carnum+1

            video_capture.release()
            cv2.destroyAllWindows()

            crash_result = divide(video_classification,crashcar[0],crashcar[1],crashcarnum[0],crashcarnum[1])
            self.show_result(crash_result)

    def show_result(self, crash_result):
        print(crash_result)
        if crash_result[0] == 0:
            case = "T자 교차로 좌(우)회전 사고"
        elif crash_result[0]==1:
            case = "사거리 교차로 좌회전 대 좌회전 사고"
        elif crash_result[0]==2:
            case = "역주행 사고"
        elif crash_result[0]==3:
            case = "이면도로 사고"
        elif crash_result[0]==4:
            case = "정차 후 출발 사고"
        elif crash_result[0]==5:
            case = "주'정차 중 추돌사고"
        elif crash_result[0]==6:
            case = "주차장 사고"
        elif crash_result[0]==7:
            case = "주행 중 추돌사고"
        elif crash_result[0]==8:
            case = "차도가 아닌 곳에서 진입사고"
        elif crash_result[0]==9:
            case = "차선변경 사고"
        #case = str(crash_result[0])
        self.whatcase.setText(case)
        rcar1 = cv2.imread(str(crash_result[1]),1)
        rratio1 = crash_result[2]
        rcar2 = cv2.imread(str(crash_result[3]),1)
        rratio2 = crash_result[4]

        rcar1 = cv2.cvtColor(rcar1, cv2.COLOR_BGR2RGB)
        rcar1 = cv2.resize(rcar1, (200,100))
        h1, w1, c1= rcar1.shape
        qImg1 = QImage(rcar1.data,w1,h1,QImage.Format_RGB888)
        pixmap1 = QPixmap(qImg1)
        self.car1.setPixmap(pixmap1)
        self.car1.update()

        rcar2 = cv2.cvtColor(rcar2, cv2.COLOR_BGR2RGB)
        rcar2 = cv2.resize(rcar2, (200,100))
        h2, w2, c2= rcar2.shape
        qImg2 = QImage(rcar2.data,w2,h2,QImage.Format_RGB888)
        pixmap2 = QPixmap(qImg2)
        self.car2.setPixmap(pixmap2)
        self.car2.update()

        self.ratio1.setText(rratio1)
        self.ratio2.setText(rratio2)
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec_()

