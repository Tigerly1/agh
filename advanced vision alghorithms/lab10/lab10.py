import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter

from materialy_feature_points.pm import appendimages, plot_matches
from os.path import join

path = './materialy_feature_points/'

def harris_corner_detector(image, sobel_mask_size, gaussian_mask_size):
    #Na początku obliczane są gradienty obrazu w poziomie (Ix) i w pionie (Iy) za pomocą operatora Sobela.
    Ix = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=sobel_mask_size)
    Iy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=sobel_mask_size)
    #Następnie obliczane są kwadraty tych gradientów (Ixx i Iyy) oraz ich iloczyn (Ixy). Każde z tych trzech obliczeń jest dodatkowo rozmywane filtrem gaussowskim.
    Ixx = cv2.GaussianBlur(Ix**2, (gaussian_mask_size, gaussian_mask_size), 0)
    Iyy = cv2.GaussianBlur(Iy**2, (gaussian_mask_size, gaussian_mask_size), 0)
    Ixy = cv2.GaussianBlur(Ix*Iy, (gaussian_mask_size, gaussian_mask_size), 0)
    
    det = Ixx * Iyy - Ixy ** 2
    trace = Ixx + Iyy
    
    k = 0.05
    harris_response = det - k * trace ** 2
    harris_response = harris_response / harris_response.max() # normalize to 0-1
    
    return harris_response

def find_max(image, size, threshold): # size - maximum filter mask size
    data_max = maximum_filter(image, size)
    maxima = (image==data_max)
    diff = image>threshold
    maxima[diff == 0] = 0
    return np.nonzero(maxima)

def plot_image_and_features(image, features):
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.plot(features[1], features[0], '*', color='r')
    plt.show()

def main1():
    fontanna_1 = join(path, 'fontanna1.jpg')
    fontanna_2 = join(path, 'fontanna2.jpg')
    # Load the images
    image1 = cv2.imread(fontanna_1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(fontanna_2, cv2.IMREAD_GRAYSCALE)

    # Apply the Harris Corner Detector
    harris_response1 = harris_corner_detector(image1, sobel_mask_size=3, gaussian_mask_size=7)
    harris_response2 = harris_corner_detector(image2, sobel_mask_size=3, gaussian_mask_size=7)

    # Find local maxima
    features1 = find_max(harris_response1, size=7, threshold=0.3)
    features2 = find_max(harris_response2, size=7, threshold=0.3)

    # Plot the images with the features
    plot_image_and_features(image1, features1)
    plot_image_and_features(image2, features2)

    fontanna_1 = join(path, 'budynek1.jpg')
    fontanna_2 = join(path, 'budynek2.jpg')
    # Load the images
    image1 = cv2.imread(fontanna_1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(fontanna_2, cv2.IMREAD_GRAYSCALE)

    # Apply the Harris Corner Detector
    harris_response1 = harris_corner_detector(image1, sobel_mask_size=3, gaussian_mask_size=7)
    harris_response2 = harris_corner_detector(image2, sobel_mask_size=3, gaussian_mask_size=7)

    # Find local maxima
    features1 = find_max(harris_response1, size=7, threshold=0.01)
    features2 = find_max(harris_response2, size=7, threshold=0.01)

    # Plot the images with the features
    plot_image_and_features(image1, features1)
    plot_image_and_features(image2, features2)


# def detect_keypoints(image, blockSize=3, ksize=5, k=0.2):
#     """
#     Wykrywanie punktów charakterystycznych za pomocą metody Harrisa.
#     """
    
#     # Wykrywanie punktów charakterystycznych za pomocą metody Harrisa
#     dst = cv2.cornerHarris(image, blockSize, ksize, k)

#     # Rozszerzenie wyniku dla lepszego zaznaczenia punktów charakterystycznych na obrazie
#     dst = cv2.dilate(dst, None)

#     # Zwróć współrzędne punktów charakterystycznych
#     keypoints = np.argwhere(dst > 0.17 * dst.max())

#     return keypoints

def describe_keypoints(image, keypoints, size):
    """
    Opisywanie punktów charakterystycznych.
    """
    # Usuwanie punktów, których otoczenia nie mieszczą się w obrazie
    height, width = image.shape[:2]
    keypoints = [pt for pt in keypoints if pt[0] >= size and pt[0] < height - size and pt[1] >= size and pt[1] < width - size]
    print("keypoints")
    print(keypoints)
    keypoints = np.transpose(keypoints)
    print("keypoints")
    print(keypoints)
    # Utworzenie listy opisów punktów
    descriptors = [image[y-size:y+size+1, x-size:x+size+1].flatten() for y, x in keypoints if image[y-size:y+size+1, x-size:x+size+1].size == (2*size+1)**2]

    return keypoints, descriptors

def compare_descriptors(descriptors1, descriptors2, n):
    """
    Porównywanie opisów punktów charakterystycznych z dwóch obrazów i znajdowanie n najbardziej podobnych.
    """
    matches = []
    matches_pm = []
    print(len(descriptors1), len(descriptors2))

    for i, d1 in enumerate(descriptors1):
        # Obliczenie odległości (np. euklidesowych) od opisu d1 do wszystkich opisów w descriptors2
        distances = [np.linalg.norm(d1-d2) for d2 in descriptors2]
        print(i)
        # Znalezienie indeksu najmniejszej odległości
        min_index = np.argmin(distances)

        # Dodanie do listy dopasowań
        matches.append(((i, min_index), distances[min_index]))
    # Sortowanie listy dopasowań względem odległości
    matches.sort(key=lambda x: x[1])
    # Zwrócenie n najlepszych dopasowań
    return matches[:n], matches_pm[:n]

def main2(image_name):
    # Wczytywanie obrazów
    image1 = join(path, image_name+'1.jpg')
    image2 = join(path, image_name+'2.jpg')    

    image1 = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2, cv2.IMREAD_GRAYSCALE)


    # Apply the Harris Corner Detector
    harris_response1 = harris_corner_detector(image1, sobel_mask_size=3, gaussian_mask_size=7)
    harris_response2 = harris_corner_detector(image2, sobel_mask_size=3, gaussian_mask_size=7)

    # Find local maxima
    keypoints1 = find_max(harris_response1, size=7, threshold=0.2)
    keypoints2 = find_max(harris_response2, size=7, threshold=0.2)
    print(keypoints1)
    # Wykrywanie punktów charakterystycznych
    # keypoints1 = detect_keypoints(image1)
    # keypoints2 = detect_keypoints(image2)
    print(1)
    # Opisywanie punktów charakterystycznych
    keypoints1, descriptors1 = describe_keypoints(image1, keypoints1, 20)
    keypoints2, descriptors2 = describe_keypoints(image2, keypoints2, 20)
    print(1)
    # Porównanie opisów punktów charakterystycznych
    matches, matches_pm = compare_descriptors(descriptors1, descriptors2, 100)
    print(1)
    # Przygotowanie listy współrzędnych punktów dla funkcji do rysowania
    points1 = np.array([keypoints1[i] for ((i, _), _) in matches])
    points2 = np.array([keypoints2[i] for ((_, i), _) in matches])
    print(1)   
    matches_pm = np.array([((keypoints1[i], keypoints2[j])) for ((i, j), _) in matches])
    print(matches_pm)
    # matches_zipped = [() for ((x,z) , y) in matches]
    # print('Liczba dopasowań:', len(matches))
    # print(points1)
    # print(matches)
    # Wyświetlanie wyników
    plot_matches(image1, image2, matches_pm)
    # plt.figure(figsize=(20, 10))

   
    # plt.subplot(121)
    # plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    # plt.scatter(points1[:, 1], points1[:, 0], c='r', s=3)

    # plt.subplot(122)
    # plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    # plt.scatter(points2[:, 1], points2[:, 0], c='r', s=3)

    plt.show()

def FAST(img, t: int = 10, n: int=9, offset: int=15):
    imgg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(int)
    Xshift = [-1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2]
    Yshift = [3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2]

    height, width = imgg.shape

    corners = []
    for x in range(offset, width-offset):
        for y in range(offset, height-offset):
            core = imgg[y][x]
            thresholds = []
            for xshift, yshift in zip(Xshift, Yshift):
                if imgg[y+yshift][x+xshift] > core+t:
                    thresholds.append(1)
                elif imgg[y+yshift][x+xshift] < core-t:
                    thresholds.append(-1)
                else:
                    thresholds.append(0)
            #print(thresholds)

            for i in range(len(thresholds)):
                count = 0
                for j in range(0, n-1):
                    if thresholds[(i+j) % n] == thresholds[(i+j+1) % n] and (thresholds[(i+j) % n] == 1 or thresholds[(i+j) %n] ==-1):
                        count +=1
                    else:
                        break
                #print("Found corner:", x, y)
                if count >= n-1:
                    corners.append((y, x))
                    break
    #print(corners)
    return corners

def HarrisFAST(img, points, sobel: int=7, gauss: int=7, n: int=20):

    ret = np.zeros_like(img).astype(float)

    sobel_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize = sobel)
    sobel_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize = sobel)

    Ixx = np.square(sobel_x)
    Iyy = np.square(sobel_y)
    Ixy = np.multiply(sobel_x, sobel_y)

    XX_blur = cv2.GaussianBlur(Ixx, (gauss, gauss), 0)
    YY_blur = cv2.GaussianBlur(Iyy, (gauss, gauss), 0)
    XY_blur = cv2.GaussianBlur(Ixy, (gauss, gauss), 0)

    K = 0.05

    M = lambda x, y: np.array([[XX_blur[x][y], XY_blur[x][y]], [XY_blur[x][y], YY_blur[x][y]]])
    H = lambda x, y: np.linalg.det(M(x, y)) - K*np.trace(M(x, y))

    values = []
    for point in points:
        value = H(point[0], point[1])
        if 0 <= point[1] < ret.shape[0] and 0 <= point[0] < ret.shape[1]:
            ret[point[1], point[0]] = value
            values.append((point, value))
    #print(len(values))

    filtered = []
    for point, value in values:
         if np.max(ret[point[1]-1:point[1]+2, point[0]-1:point[0]+2]) == value:
             filtered.append((point, value))
    #print(len(filtered))
    filtered.sort(key=lambda x: x[1], reverse=True)
    filtered = filtered[:n]
    #print(len(filtered))
    return filtered

def brightness_centroid(img, points, size: int=15):
    ret = []
    for point, value in points:
        w = img[point[0]-size:point[0]+size+1, point[1]-size:point[1]+size+1]

        circle = np.zeros_like(w)
        circle = cv2.circle(circle, (size, size), size+1, 1, -1)
        w = cv2.bitwise_and(w, circle)

        m = cv2.moments(w)
        cx = int(m['m10']/m['m00']) - size + point[0]
        cy = int(m['m01']/m['m00']) - size + point[1]
        c = (cx, cy)

        theta = np.arctan2(m['m01'], m['m10'])

        ret.append((point, value, c, theta))
    return ret

def BRIEF(img, points, pairs, size: int=15):
    ret = []

    for (x, y), value, c, theta in points:

        xp1 = np.cos(theta)*pairs[:, 0] - np.sin(theta)*pairs[:, 1]
        yp1 = np.cos(theta)*pairs[:, 0] + np.sin(theta)*pairs[:, 1]

        xp2 = np.cos(theta)*pairs[:, 2] - np.sin(theta)*pairs[:, 3]
        yp2 = np.cos(theta)*pairs[:, 2] + np.sin(theta)*pairs[:, 3]

        transformed = np.transpose(np.array([xp1, yp1, xp2, yp2]))
        transformed = transformed.astype(int)

        hamming = []

        for xp1, yp1, xp2, yp2 in transformed:
            hamming.append(img[x+xp1, y+yp1] < img[x+xp2, y+yp2])

        ret.append(((x,y), value, c, theta, hamming))
    return ret

def draw_points(img, h):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 'gray')
    plt.plot(h[1], h[0], '*', color='r')

def ORB(img1, img2, threshold, pairs, n: int = 20):

    f1 = FAST(img1, threshold)
    f2 = FAST(img2, threshold)

    f1g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    f2g = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    f1h = HarrisFAST(f1g, f1)
    f2h = HarrisFAST(f2g, f2)
    fastharris_coords1 = [x[0] for x in f1h]
    fastharris_coords2 = [x[0] for x in f2h]
    points1 = tuple([tuple([item[i] for item in fastharris_coords1]) for i in range(2)])
    points2 = tuple([tuple([item[i] for item in fastharris_coords2]) for i in range(2)])

    plt.figure()
    draw_points(img1, points1)
    plt.figure()
    draw_points(img2, points2)

    f1c = brightness_centroid(f1g, f1h)
    f2c = brightness_centroid(f2g, f2h)

    brief1 = BRIEF(f1g, f1c, pairs)
    brief2 = BRIEF(f2g, f2c, pairs)

    pairs = []
    for i in range(len(brief1)):
        for j in range(len(brief2)):
            pairs.append((brief1[i][:-1], brief2[j][:-1], np.sum(np.logical_xor(brief1[i][4], brief2[j][4]).astype(int))))
    pairs.sort(key=lambda x: x[2])
    pairs = pairs[:n]

    return pairs

def main3(image_name = 'fontanna'):
    # Wczytanie obrazów
    image1 = join(path, image_name+'1.jpg')
    image2 = join(path, image_name+'2.jpg')    

    image1 = cv2.imread(image1, )
    image2 = cv2.imread(image2, )

    # Inicjalizacja detektora ORB
    orb = cv2.ORB_create()

    # Wykrywanie punktów kluczowych i obliczanie deskryptorów
    kp1, des1 = orb.detectAndCompute(image1,None)
    kp2, des2 = orb.detectAndCompute(image2,None)

    # Dopasowanie punktów kluczowych za pomocą metryki Hamminga
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1,des2)

    # Sortowanie dopasowań według odległości.
    matches = sorted(matches, key = lambda x:x.distance)

    # Wyświetlanie N pierwszych dopasowań
    N = 1000
    matching_result = cv2.drawMatches(image1, kp1, image2, kp2, matches[:N], None, flags=2)

    cv2.imshow("Matching result", matching_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main4(image_name = 'fontanna'):
    

    # 1. Wczytaj obrazy i przekonwertuj do skali szarości
    image_left = cv2.imread(join(path, 'left_panorama.jpg'), cv2.IMREAD_GRAYSCALE)
    image_right = cv2.imread(join(path, 'right_panorama.jpg'), cv2.IMREAD_GRAYSCALE)

    # 2. Znajdź punkty charakterystyczne na obrazach
    sift = cv2.SIFT_create()
    keypointsL, descriptorsL = sift.detectAndCompute(image_left, None)
    keypointsR, descriptorsR = sift.detectAndCompute(image_right, None)

    # 3. Wyświetl znalezione punkty
    image_left_keypoints = cv2.drawKeypoints(image_left, keypointsL, None)
    image_right_keypoints = cv2.drawKeypoints(image_right, keypointsR, None)
    cv2.imshow("Left keypoints", image_left_keypoints)
    cv2.imshow("Right keypoints", image_right_keypoints)

    # 4. Dopasuj punkty charakterystyczne z obu obrazów
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(descriptorsL, descriptorsR, k=2)

    # 5. & 6. Użyj KNN i wybierz najlepsze dopasowania
    best_matches = [[m] for m, n in matches if m.distance < 0.5 * n.distance]

    # 7. Wyświetl otrzymane połączenia
    matched_image = cv2.drawMatchesKnn(image_left, keypointsL, image_right, keypointsR, best_matches, None, flags=2)
    cv2.imshow("Matches", matched_image)

    # 8. Określ rotację i translację obrazów
    ptsA = np.float32([keypointsL[m[0].queryIdx].pt for m in best_matches])
    ptsB = np.float32([keypointsR[m[0].trainIdx].pt for m in best_matches])
    H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC)

    # 9. Transformuj obraz
    height = image_left.shape[0] + image_right.shape[0]
    width = image_left.shape[1] + image_right.shape[1]    
    result = cv2.warpPerspective(image_left, H, (width, height))
    result[0:image_right.shape[0], 0:image_right.shape[1]] = image_right

    # 10. Wyświetl wyniki
    cv2.imshow("Stitched image", result)

    # 11. Usuń nadmiar czarnego tła
    if len(result.shape) == 3:  # Obraz kolorowy, więc przekształć go do skali szarości
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    else:  # Obraz jest już w skali szarości
        gray = result
    _, threshold = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    cropped = result[y:y+h, x:x+w]
    cv2.imshow("Cropped image", cropped)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    #main1()
    #main2('fontanna')
    #main3()
    main4()
