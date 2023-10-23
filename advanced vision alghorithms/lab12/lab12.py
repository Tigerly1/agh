import numpy as np
from scipy.ndimage import filters
import cv2
import math
from sklearn import svm
from sklearn.metrics import confusion_matrix
from os.path import join
from imutils.object_detection import non_max_suppression
from sklearn.model_selection import train_test_split
from scipy import ndimage
path = './pedestrians/'


def calculate_gradient(img):
    gradient_x = ndimage.convolve1d(np.int32(img), np.array([-1, 0, 1]), 1)
    gradient_y = ndimage.convolve1d(np.int32(img), np.array([-1, 0, 1]), 0)
    magnitude = np.hypot(gradient_x, gradient_y)
    orientation = np.rad2deg(np.arctan2(gradient_y, gradient_x)) % 180
    return magnitude, orientation

def cell_histogram(cell_magnitude, cell_angle, bin_size):
    bins_range = np.int32(np.linspace(0, 180, bin_size, endpoint=False))
    digitized = np.digitize(cell_angle, bins_range, right=True)
    bins = np.zeros(bin_size)
    for i in range(bin_size):
        bins[i] = np.sum(cell_magnitude[digitized == i+1])
    return bins

def calculate_histograms(magnitude, orientation, cell_size=8, bin_size=9):
    if len(magnitude.shape) == 3:
        magnitude = cv2.cvtColor(magnitude, cv2.COLOR_BGR2GRAY)
        orientation = cv2.cvtColor(orientation, cv2.COLOR_BGR2GRAY)
    height, width = magnitude.shape

    cell_rows, cell_cols = height // cell_size, width // cell_size
    histogram = np.zeros((cell_rows, cell_cols, bin_size))
    for i in range(cell_rows):
        for j in range(cell_cols):
            cell_magnitude = magnitude[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            cell_angle = orientation[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            histogram[i, j] = cell_histogram(cell_magnitude, cell_angle, bin_size)
    return histogram

def normalize_histogram(histogram, eps=1e-5):
    height, width, _ = histogram.shape
    normalized_histogram = np.zeros((height-1, width-1, 36))
    for i in range(height-1):
        for j in range(width-1):
            block = histogram[i:i+2, j:j+2].ravel()
            normalized_histogram[i, j] = block / np.sqrt(np.linalg.norm(block)**2 + eps**2)
    return normalized_histogram

def HOG(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    magnitude, orientation = calculate_gradient(img)
    histogram = calculate_histograms(magnitude, orientation)
    normalized_histogram = normalize_histogram(histogram)
    return normalized_histogram.ravel()

# Usage:
img = cv2.imread(join(path, 'pos/per00060.ppm'), cv2.IMREAD_GRAYSCALE)
features = HOG(img)
print(features[:10])

# Przygotowanie danych
HOG_data = np.zeros([2*900,3781],np.float32)
for i in range(0,900):
    IP = cv2.imread(join(path,f'pos/per{i+1:05}.ppm'))
    IN = cv2.imread(join(path,f'neg/neg{i+1:05}.png'))
    F = HOG(IP)
    HOG_data[i,0] = 1
    HOG_data[i,1:] = F
    F = HOG(IN)
    HOG_data[i+900,0] = 0
    HOG_data[i+900,1:] = F

# Podział na etykiety i dane
labels = HOG_data[:,0]
data = HOG_data[:,1:]

# Podział na zestawy treningowe i testowe
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, shuffle=True)

# Tworzenie klasyfikatora SVM
clf = svm.SVC(kernel='linear', C=1.0)

# Uczenie klasyfikatora
clf.fit(data_train, labels_train)

# Predykcja na danych testowych
labels_pred = clf.predict(data_test)

# Analiza wyników uczenia
TN, FP, FN, TP = confusion_matrix(labels_test, labels_pred).ravel()
print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")

# Predykcja na danych testowych
labels_pred = clf.predict(data_test)

# Analiza wyników uczenia
TN, FP, FN, TP = confusion_matrix(labels_test, labels_pred).ravel()
print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")

def sliding_window(image, stepSize, windowSize):
    # przesuń okno na obrazie
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield koordynaty bieżącego okna
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

# Wczytaj obraz testowy
image = cv2.imread('./test_pedestrians/testImage4.png')
# Definiujemy rozmiar okna
(winW, winH) = (64, 128)

detections = []

def pyramid(image, scale=1.5, minSize=(128, 256)):
    # Zwraca obraz wraz z kolejnymi skalami
    yield image, scale

    # Kontynuuj skalowanie dopóki nie osiągniesz minimalnego rozmiaru
    while True:
        # Oblicz nowy rozmiar obrazu i skaluj go
        w = int(image.shape[1] / scale)
        h = int(image.shape[0] / scale)
        image = cv2.resize(image, (w, h))
        # Jeśli obraz jest mniejszy niż minimalny rozmiar, zakończ skalowanie
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        # Zwróć obraz
        yield image, scale

# Przeskaluj obraz i wykonaj przesuwające się okno
for resized, scale in pyramid(image, scale=1.5):
    for (x, y, window) in sliding_window(image, stepSize=8, windowSize=(winW, winH)):
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        # Wyliczamy deskryptor HOG dla okna
        F = HOG(window)

        # Klasyfikujemy okno jako pieszy lub nie
        pred = clf.predict([F])
        
        # Jeśli okno jest klasyfikowane jako pieszy, zapisujemy jego pozycję
        if pred == 1:
            detections.append((x, y, clf.decision_function([F]),
                                int(winW*scale),
                                int(winH*scale)))
        
scaled_detections = [(x, y, _x, _y) for (x, y, _, _x, _y) in detections]

print(detections)
# Wykorzystaj Non-Maximum Suppression do wyeliminowania nakładających się detekcji
pick = non_max_suppression(np.array(scaled_detections), probs=None, overlapThresh=0.3)

for (xA, yA, xB, yB) in pick:
    # Rysowanie prostokąta
    cv2.rectangle(image, (xA, yA), (xA + xB, yA + yB), (0, 255, 0), 2)

cv2.imshow("Detections", image)
cv2.waitKey(0)