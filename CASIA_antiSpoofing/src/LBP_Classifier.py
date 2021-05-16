import cv2
import os
import FeatureExtraction as ft
import numpy as np
from sklearn.externals import  joblib
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

def detect_faces(face_detector, img):
    detected_faces = []
    faces = face_detector.detectMultiScale(img, 1.3, 5)
    for (x, y, w, h) in faces:
        detected_faces.append(cv2.resize(img[y:y+h, x:x+w], (110, 110)))

    return detected_faces


def get_features_and_labels(data_list_path):
    face_detector = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')

    X = []
    # let [1, 0] represent REAL and [0, 1] represent MASK
    REAL = np.array([1, 0])
    MASK = np.array([0, 1])
    y = []
    data_list = os.listdir(data_list_path)
    for a_persons_video in data_list:
        a_persons_video_path = data_list_path + '/' + a_persons_video
        # dir_list_for_REAL.append(a_persons_video_path + '/' + 'HR_1.avi')
        # maybe we will end up looping this separately with the lists of dirs from real and mask vids
        i = 0  # i is the frame counter
        cap = cv2.VideoCapture(a_persons_video_path + '/' + 'HR_1.avi')
        while True:
            continue_video_loop, frame = cap.read()
            if not continue_video_loop:
                break
            else:
                print('the', '[', i, ']', 'th', 'frame in', a_persons_video_path + '/' + 'HR_1.avi')
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # detected_faces = detect_faces(face_detector, gray_frame)
                # if len(detected_faces) == 0:
                #    print('No face in this frame!')

                # else:
                lbp = ft.LBP(8, 1)
                lbp_histogram = ft.getFeatureVector(gray_frame, lbp)
                print(lbp_histogram.shape)
                X.append(lbp_histogram)
                y.append(REAL)
            i += 1
        cap.release()
        # -----------------------------handle the masks---------------------------
        cap = cv2.VideoCapture(a_persons_video_path + '/' + 'HR_2.avi')
        while True:
            continue_video_loop, frame = cap.read()
            if not continue_video_loop:
                break
            else:
                print('the', '[', i, ']', 'th', 'frame in', a_persons_video_path + '/' + 'HR_2.avi')
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # detected_faces = detect_faces(face_detector, gray_frame)
                # if len(detected_faces) == 0:
                #    print('No face in this frame!')

                # else:
                lbp = ft.LBP(8, 1)
                lbp_histogram = ft.getFeatureVector(gray_frame, lbp)
                print(lbp_histogram.shape)
                X.append(lbp_histogram)
                y.append(MASK)
            i += 1
        cap.release()

    X = np.vstack(tuple(X))
    print('X shape now', X.shape)
    y = np.vstack(tuple(y))
    print('y shape now', y.shape)
    return X, y


def save_features_and_labels(data_list_path, forTraining=True):
    if forTraining:
        X, y = get_features_and_labels(data_list_path)
        os.makedirs('data/training/', exist_ok=True)
        joblib.dump({'X': X, 'y': y}, 'data/training/X_and_y_for_training.pkl')
    else:
        X, y = get_features_and_labels(data_list_path)
        os.makedirs('data/testing/', exist_ok=True)
        joblib.dump({'X': X, 'y': y}, 'data/testing/X_and_y_for_testing.pkl')


def load_features_and_labels(forTraining=True):
    if forTraining:
        dataSet = joblib.load('data/training/X_and_y_for_training.pkl')
        return dataSet['X'], dataSet['y']
    else:
        dataSet = joblib.load('data/testing/X_and_y_for_testing.pkl')
        return dataSet['X'], dataSet['y']

def save_classifier(clf, filename):
    joblib.dump(clf, filename)


def load_classifier(filename):
    return joblib.load(filename)

def fix_shape_of_y(y):
    new_y = np.zeros((y.shape[0], ))
    for i in range(y.shape[0]):
        #print(y[i,:][0], y[i,:][1])
        if y[i, :][0] == 1 and y[i, :][1] == 0:
            new_y[i] = 1
        elif y[i, :][0] == 0 and y[i, :][1] == 1:
            new_y[i] = 0
        else:
            print('play with dimensions')
    return new_y
if __name__ == '__main__':
    '''
    data_set_path = 'data/CASIA-FA'
    face_data_list = os.listdir(data_set_path)  # ['test_release', 'train_release']
    training_data_list_path = data_set_path + '/' + face_data_list[1]
    testing_data_list_path = data_set_path + '/' + face_data_list[0]
    print('save traininng  and testing features and labels')
    save_features_and_labels(training_data_list_path, forTraining=True)
    save_features_and_labels(training_data_list_path, forTraining=False)

'''
    print('get features and labels for training and testing')
    X_training, y_training = load_features_and_labels(forTraining=True)
    y_training = fix_shape_of_y(y_training)
    print(y_training.shape)
    X_testing, y_testing = load_features_and_labels(forTraining=False)
    y_testing = fix_shape_of_y(y_testing)
    print(y_testing.shape)
    '''
    print('SVM:')
    sv = svm.SVC(probability=True)
    print('train classifier')
    sv.fit(X_training, y_training)
    print('save the trained classifier')
    save_classifier(sv, 'svm_FASD.pkl')
    #'''
    print('K Nearest Neighbors:')
    knn = KNeighborsClassifier(n_neighbors=2)
    print('train classifier')
    knn.fit(X_training, y_training)
    print('save the trained classifier')
    save_classifier(knn, 'knn_FASD.pkl')
    print('------------------------------------------')

    print('K Nearest Neighbors:')
    knn = KNeighborsClassifier(n_neighbors=2)
    print('train classifier')
    knn.fit(X_training, y_training)
    print('save the trained classifier')
    save_classifier(knn, 'knn_FASD.pkl')
    print('------------------------------------------')

    print('Gaussian Naive Bayes:')
    gnb = GaussianNB()
    print('train classifier')
    gnb.fit(X_training, y_training)
    print('save the trained classifier')
    save_classifier(gnb, 'gnb_FASD.pkl')
    print('------------------------------------------')

    print('Random Forest Classifier:')
    rf = RandomForestClassifier()
    print('train classifier')
    rf.fit(X_training, y_training)
    print('save the trained classifier')
    save_classifier(rf, 'rf_FASD.pkl')
    print('------------------------------------------')

    print('Logistic Regression:')
    lr = LogisticRegression()
    print('train classifier')
    lr.fit(X_training, y_training)
    print('save the trained classifier')
    save_classifier(lr, 'lr_FASD.pkl')
    print('------------------------------------------')

    print('------------DONE TRAINING CLASSIFIERS--------------')
    '''
    loaded_classifier = load_classifier('svm_FASD.pkl')
    print(type(loaded_classifier))
    predictions = loaded_classifier.predict(X_testing)
    print(confusion_matrix(y_testing, predictions))
    print('---------------------')
    print(accuracy_score(y_testing, predictions))
    '''
