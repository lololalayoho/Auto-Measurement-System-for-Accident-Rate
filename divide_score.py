from sklearn.preprocessing import StandardScaler
from sklearn import svm
import joblib
import cv2
def divide(video_classification,car1,car2,car1_num,car2_num):
    max_index=video_classification.index(max(video_classification))
    cnt1 = 0
    cnt2 = 0
    label1 = 0
    label2 = 0
    if max_index==0:
        clf = joblib.load('randomForest/T_road.pkl')
        if len(car1) > len(car2):
            while True:
                car1.pop()
                
                if len(car1)==len(car2):
                    break
        else:
            while True:
                if len(car1)==len(car2):
                    break
                else:
                    car2.pop()
        pre1 = clf.predict(car1)
        pre2 = clf.predict(car2)
        for i in range(len(pre1)):
            if pre1[i]==1:
                cnt1 = cnt1 + 1
        for j in range(len(pre2)):
             if pre2[j]==1:
                 cnt2 = cnt2+1
        if cnt1 > cnt2:
            label1 = 70
            label2 = 30
        else:
            label1 = 30
            label2 = 70
    elif max_index==1:
        clf = joblib.load('randomForest/Four.pkl')
        if len(car1) > len(car2):
            while True:
                car1.pop()
                if len(car1)==len(car2):
                    break
        else:
            while True:
                if len(car1)==len(car2):
                    break
                else:
                    car2.pop()
        pre1 = clf.predict(car1)
        pre2 = clf.predict(car2)
        for i in range(len(pre1)):
            if pre1[i] == 1:
                cnt1 = cnt1 + 1
        for j in range(len(pre2)):
            if pre2[j] == 1:
                cnt2 = cnt2 + 1
        if cnt1 > cnt2:
            label1 = 40
            label2 = 60
        else:
            label1 = 60
            label2 = 40
    elif max_index == 2:
        clf = joblib.load('randomForest/reverse.pkl')
        if len(car1) > len(car2):
            while True:
                car1.pop(0)
                if len(car1)==len(car2):
                    break
        else:
            while True:
                if len(car1)==len(car2):
                    break
                else:
                    car2.pop(0)
        pre1 = clf.predict(car1)
        pre2 = clf.predict(car2)
        for i in range(len(pre1)):
            if pre1[i] == 1:
                cnt1 = cnt1 + 1
        for j in range(len(pre2)):
            if pre2[j] == 1:
                cnt2 = cnt2 + 1
        if cnt1 > cnt2:
            label1 = 0
            label2 = 100
        else:
            label1 = 100
            label2 = 0
    elif max_index == 3:
            label1 = 50
            label2 = 50
    elif max_index == 4:
        clf = joblib.load('randomForest/Stop_and_Go.pkl')
        if len(car1) > len(car2):
            while True:
                car1.pop()
                if len(car1)==len(car2):
                    break
        else:
            while True:
                if len(car1)==len(car2):
                    break
                else:
                    car2.pop()
        pre1 = clf.predict(car1)
        pre2 = clf.predict(car2)
        for i in range(len(pre1)):
            if pre1[i] == 1:
                cnt1 = cnt1 + 1
        for j in range(len(pre2)):
            if pre2[j] == 1:
                cnt2 = cnt2 + 1
        if cnt1 > cnt2:
            label1 = 20
            label2 = 80
        else:
            label1 = 80
            label2 = 20
    elif max_index == 5:
        clf = joblib.load('randomForest/Stop_accident.pkl')
        if len(car1) > len(car2):
            while True:
                car1.pop(0)
                if len(car1)==len(car2):
                    break
        else:
            while True:
                if len(car1)==len(car2):
                    break
                else:
                    car2.pop(0)
        pre2 = clf.predict(car2)
        pre1 = clf.predict(car1)
        for i in range(len(pre1)):
            if pre1[i] == 1:
                cnt1 = cnt1 + 1
        for j in range(len(pre2)):
            if pre2[j] == 1:
                cnt2 = cnt2 + 1
        if cnt1 > cnt2:
            label1 = 100
            label2 = 0
        else:
            label1 = 0
            label2 = 100
    elif max_index == 6:
        clf = joblib.load('randomForest/ParkingLot.pkl')
        if len(car1) > len(car2):
            while True:
                car1.pop()
                if len(car1)==len(car2):
                    break
        else:
            while True:
                if len(car1)==len(car2):
                    break
                else:
                    car2.pop()
        pre1 = clf.predict(car1)
        pre2 = clf.predict(car2)
        for i in range(len(pre1)):
            if pre1[i] == 1:
                cnt1 = cnt1 + 1
        for j in range(len(pre2)):
            if pre2[j] == 1:
                cnt2 = cnt2 + 1
        if cnt1 > cnt2:
            label1 = 25
            label2 = 75
        else:
            label1 = 75
            label2 = 25
    elif max_index == 7:
        clf = joblib.load('randomForest/driving_accident.pkl')
        if len(car1) > len(car2):
            while True:
                car1.pop(0)
                if len(car1)==len(car2):
                    break
        else:
            while True:
                if len(car1)==len(car2):
                    break
                else:
                    car2.pop(0)
        pre1 = clf.predict(car1)
        pre2 = clf.predict(car2)
        for i in range(len(pre1)):
            if pre1[i] == 1:
                cnt1 = cnt1 + 1
        for j in range(len(pre2)):
            if pre2[j] == 1:
                cnt2 = cnt2 + 1
        if cnt1 > cnt2:
            label1 = 100
            label2 = 0
        else:
            label1 = 0
            label2 = 100
    elif max_index == 8:
        clf = joblib.load('randomForest/Not_Road.pkl')
        if len(car1) > len(car2):
            while True:
                car1.pop()
                if len(car1)==len(car2):
                    break
        else:
            while True:
                if len(car1)==len(car2):
                    break
                else:
                    car2.pop()
        pre1 = clf.predict(car1)
        pre2 = clf.predict(car2)
        for i in range(len(pre1)):
            if pre1[i] == 1:
                cnt1 = cnt1 + 1
        for j in range(len(pre2)):
            if pre2[j] == 1:
                cnt2 = cnt2 + 1
        if cnt1 > cnt2:
            label1 = 20
            label2 = 80
        else:
            label1 = 80
            label2 = 20
    elif max_index == 9:
        clf = joblib.load('randomForest/change_Road.pkl')
        if len(car1) > len(car2):
            while True:
                car1.pop(0)
                if len(car1)==len(car2):
                    break
        else:
            while True:
                if len(car1)==len(car2):
                    break
                else:
                    car2.pop(0)
        pre1 = clf.predict(car1)
        pre2 = clf.predict(car2)
        for i in range(len(pre1)):
            if pre1[i] == 1:
                cnt1 = cnt1 + 1
        for j in range(len(pre2)):
            if pre2[j] == 1:
                cnt2 = cnt2 + 1
        if cnt1 > cnt2:
            label1 = 30
            label2 = 70
        else:
            label1 = 70
            label2 = 30
    name1 = 'car' + str(car1_num) + '.jpg'
    name2 = 'car' + str(car2_num) + '.jpg'
    accident_result =[]
    accident_result.append(max_index) 
    accident_result.append(name1)
    accident_result.append(str(label1))
    accident_result.append(name2)
    accident_result.append(str(label2))

    return accident_result
'''
    image1 = cv2.imread(name1)
    image2 = cv2.imread(name2)
    cv2.putText(image1,str(label1),(40,120),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),3)
    cv2.putText(image2,str(label2),(40,120), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
    cv2.imshow(str(label1),image1)
    cv2.imwrite(str(label1)+".jpg",image1)
    cv2.imshow(str(label2),image2)
    cv2.imwrite(str(label2)+".jpg",image2)
    cv2.waitKey(0)
'''
