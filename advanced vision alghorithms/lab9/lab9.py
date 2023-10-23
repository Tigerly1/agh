# Zadanie 9.1 Wyszukiwanie wzorców.
# 1. Ze strony kursu pobierz archiwum z danymi do cwiczenia i rozpakuj je we własnym ´
# katalogu roboczym.
# 2. Utwórz nowy skrypt. Na podstawie obrazu ze wzorcem trybik.jpg stworzymy tablic˛e
# R-table. W tym celu wyznacz kontury oraz gradienty na obrazie wzorca.
# 3. Aby wyznaczyc kontur ´ nalezy najpierw przeprowadzi ˙ c konwersj˛e obrazu na odcie ´ n´
# szarosci oraz binaryzacj˛e z odpowiednim progiem. Nast˛epnie wykorzystaj funkcj˛e ´
# cv2.findContours do uzyskania konturów wyst˛epuj ˛acych na obrazie.
# Wskazówka
# Zaneguj obraz – cv2.bitwise_not – przed przesłaniem go do funkcji, gdyz˙
# zawiera on czarny kontur na białym tle, a funkcja findContours oczekuje
# odwrotnego ustawienia
# Zwróc uwag˛e, aby uzyska ´ c listy wszystkich punktów konturu – zastosuj parametr ´
# CHAIN_APPROX_NONE. Sugerowane wywołanie:
# contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.
# CHAIN_APPROX_NONE)
# Wynikiem funkcji jest m.in. lista konturów (teoretycznie na obrazie moze by ˙ c wi˛ecej ´
# obiektów) – w naszym wypadku powinna to byc lista jednoelementowa. Zdarza si˛e ´
# jednak, ze kontury si˛e podziel ˛a – wówczas rozwi ˛aza ˙ n jest kilka. Po pierwsze mo ´ zna ˙
# zmniejszyc obiekt przez erozj˛e co zapobiegnie podziałom. Ewentualnie mo ´ zna wybie- ˙
# rac najdłu ´ zszy kontur. Najpro ˙ sciej wykorzysta ´ c przy znajdowaniu konturu parametr ´
# RETR_TREE – wówczas kontury s ˛a ułozone w hierarchi˛e i najdłu ˙ zszy powinien by ˙ c´
# 72 Uogólniona transformata Hougha
# pierwszy (posiadac indeks 0). ´
# 4. Uzyskany kontur mozna nanie ˙ s´c na dowolny obraz funkcj ˛a: ´
# cv2.drawContours(img, contours, 0, color)},
# gdzie: image to obraz docelowy, contours – lista konturów, 0 – numer konturu,
# color – jasnos´c lub krotka ze składowymi koloru (w zale ´ zno ˙ sci od typu obrazu). ´
# 5. Do wyliczenia gradientów mozna wykorzysta ˙ c filtry Sobela (dla obrazu w odcieniach ´
# szarosci), przykładowo: ´
# sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
# sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
# Potrzebujemy dwie macierze zawieraj ˛ace wartosci amplitudy gradientu (jest to (pier- ´
# wiastek sumy kwadratów Sobela pionowego i poziomego) oraz orientacj˛e gradientu
# (wykorzystuj ˛ac funkcj˛e np.arctan2). Macierz wartosci amplitudy gradientu warto ´
# znormalizowac dziel ˛ac j ˛a przez jej warto ´ s´c maksymaln ˛a ( ´ np.amax()).
# 6. Przed wypełnieniem R-table nalezy˙ wybrac punkt referencyjny ´ – niech b˛edzie to
# srodek ci˛e ´ zko ˙ sci wzorca wyznaczany ze zbinaryzowanego obrazu wzorca z wykorzy- ´
# staniem momentów. Do wyznaczenia srodka ci˛e ´ zko ˙ sci mo ´ zna wykorzysta ˙ c momenty ´
# centralne m00, m10 i m01 (funkcja cv2.moments(bin, 1)).
# 7. Do wypełnienia R-table b˛ed ˛a potrzebne wektory ł ˛acz ˛ace punkty konturu/konturów
# z punktem referencyjnym. Do R-table wpisujemy długosci tych wektorów oraz k ˛aty ´
# jakie tworz ˛a z osi ˛a OX (tu znów przyda si˛e funkcja np.arctan2). Miejsce wpisania
# do tablicy R-table wyznacza orientacja gradientu w punkcie konturu (wyznaczona
# na podstawie filtracji maskami Sobela), przy czym prosz˛e przeliczyc radiany na ´
# stopnie – R-table b˛edzie miała 360 wierszy. Stosujemy dokładnos´c wynosz ˛ac ˛a 1 ´
# stopien.´ R-table mozna zaimplementowa ˙ c jako list˛e 360 list: ´
# Rtable = [[] for _ in range(360)]
# Wówczas np. Rtable[30] b˛edzie list ˛a współrz˛ednych biegunowych punktów konturu,
# których orientacja wyliczona na podstawie gradientów wynosi około 30◦
# .
# 8. Na podstawie obrazu trybiki2.jpg oraz R-table z poprzedniego punktu wypełnij dwuwymiarow ˛a przestrzen Hougha – wylicz ponownie gradient w ka ´ zdym punkcie ˙
# i dla punktów, których znormalizowana wartos´c gradientu przekracza ´ 0.5, okresl orien- ´
# tacj˛e w tym punkcie, a nast˛epnie zwi˛eksz wartos´c o 1 w przestrzeni akumulacyjnej ´
# w punktach zapisanych w R-table według zalezno ˙ sci: ´
# x1 = -r*np.cos(fi) + x
# y1 = -r*np.sin(fi) + y
# gdzie r, f i – wartosci z odpowiedniego wiersza w ´ R-table, x,y – współrz˛edne punktu,
# dla którego gradient przekracza 0.5.
# Warto tez sobie wy ˙ swietli ´ c posta ´ c przestrzeni Hougha. ´
# 9. Wyszukaj maksimum w przestrzeni Hougha i zaznacz je na obrazie wejsciowym – ´
# trybiki2.jpg. Wyznaczanie współrz˛ednych maksimum mozna przeprowadzi ˙ c za ´
# pomoc ˛a funkcji np.argmax lub konstrukcj ˛a:
# np.where(hough.max() == hough)
# gdzie: hough jest typu np.array.
# Z kolei do zaznaczania na obrazie mozna wykorzysta ˙ c funkcj˛e: ´
# plt.plot([m_x], [m_y],’*’, color=’r’))
# 73
# Natomiast jesli nie chcemy u ´ zywa ˙ c´ pyplot to mozna po prostu wyrysowa ˙ c na obrazie ´
# kółko:
# cv2.circle(I,(int(M[1]),int(M[0])),2,(0,0,255))
# gdzie: I to obraz, M to współrz˛edne znalezionego obiektu, 2 – promien, a ostatni ´
# parametr to kolor.
# 10. Wynikiem powyzszego algorytmu jest zaznaczenie znalezionego maksimum na obrazie ˙
# oraz nałozenie konturu wzorca wokół tego punktu. Przykładowe rozwi ˛azanie zostało ˙
# przedstawione na rys. 9.1, dodatkowo na rys. 9.2 zilustrowano przestrzen Hougha.

import cv2
from matplotlib import pyplot as plt
import numpy as np

# Wczytanie obrazu wzorca
base_trybik = cv2.imread('trybik.jpg')
pattern = cv2.cvtColor(base_trybik, cv2.COLOR_BGR2GRAY)
pattern = cv2.bitwise_not(pattern)
# Binaryzacja obrazu wzorca
_, bin_pattern = cv2.threshold(pattern, 100, 255, cv2.THRESH_BINARY)
bin_pattern = cv2.medianBlur(bin_pattern, 5)

# Wyszukiwanie konturów
contours, _ = cv2.findContours(bin_pattern, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

cv2.drawContours(base_trybik, [contours[0]], 0, (255, 0, 0)),
cv2.imshow('Pattern', base_trybik)
# Obliczenie momentów obrazu
moments = cv2.moments(bin_pattern)
cx = int(moments['m10'] / moments['m00'])
cy = int(moments['m01'] / moments['m00'])

# Obliczenie gradientów
sobelx = cv2.Sobel(pattern, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(pattern, cv2.CV_64F, 0, 1, ksize=5)
amp = np.sqrt(sobelx**2 + sobely**2)
amp = amp / amp.max()

orient = np.rad2deg(np.arctan2(sobely, sobelx))
orient += 180
orient = np.uint16(orient)

# Tworzenie R-table
Rtable = [[] for _ in range(360)]

for point in contours[0]:
    omega = orient[point[0, 1], point[0, 0]]
    r = np.sqrt((point[0, 0] - cx) ** 2 + (point[0, 1] - cy) ** 2)
    beta = np.arctan2(point[0, 1] - cy, point[0, 0] - cx)
    if omega == 360:
        omega = 0
    Rtable[omega].append([r, beta])

# Wczytanie obrazu do wyszukiwania
search_img = cv2.imread('trybiki2.jpg')
search_img = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
# Obliczenie gradientów
sobelx2 = cv2.Sobel(search_img, cv2.CV_64F, 1, 0, ksize=5)
sobely2 = cv2.Sobel(search_img, cv2.CV_64F, 0, 1, ksize=5)
amp2 = np.sqrt(sobelx2**2 + sobely2**2)
amp2 = amp2 / amp2.max()
orientation2 = np.rad2deg(np.arctan2(sobely2, sobelx2))
orientation2 += 180
orientation2 = np.uint16(orientation2)

# Tworzenie przestrzeni Hougha
hough_space = np.zeros(search_img.shape[:2], dtype=np.uint8)

# Wypełnianie przestrzeni Hougha
for y in range(amp2.shape[0]):
    for x in range(amp2.shape[1]):
        if amp2[y, x] > 0.5:
            
            table = Rtable[orientation2[y, x]]
            for t in table:
                x1 = int(x + t[0]*np.cos(t[1]))
                y1 = int(y + t[0]*np.sin(t[1]))
                if 0 <= x1 < amp2.shape[1] and 0 <= y1 < amp2.shape[0]:
                    hough_space[y1, x1] += 1

# Wyszukiwanie maksimum w przestrzeni Hougha
max_y, max_x = np.unravel_index(np.argmax(hough_space), hough_space.shape)
print(hough_space)


sorted_indices = np.argsort(hough_space.flatten())[::-1]

# get the top 5 maxima indices
top_5_indices = sorted_indices[:10]

# convert the indices back to 2D coordinates
top_5_maxima_coords = np.unravel_index(top_5_indices, hough_space.shape)

# get separate x and y coordinates
maxima_y, maxima_x = top_5_maxima_coords[0], top_5_maxima_coords[1]
print(maxima_x, maxima_y)

# max_coords = np.where(hough_space == hough_space.max())  # zwraca tuple
# maxima_y, maxima_x = max_coords[0], max_coords[1]  # współrzędne maxima
# print(maxima_x, maxima_y)
# 10. Nałóż kontury wzorca wokół punktów maksimum na obrazie trybiki2.jpg
contour_coords = contours[0][:, 0, :]  # współrzędne konturu
target_image = cv2.cvtColor(search_img, cv2.COLOR_GRAY2BGR)

for max_x, max_y in zip(maxima_x, maxima_y):
    # przesuwamy kontur do punktu maksimum
    new_contour_coords = contour_coords.copy()
    new_contour_coords[:, 0] += max_x - cx
    new_contour_coords[:, 1] += max_y - cy

    # zaznaczamy kontur na obrazie
    cv2.drawContours(target_image, [new_contour_coords], -1, (0, 255, 0), 2)

cv2.imshow('Detected cogs', target_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.plot([max_x], [max_y],'*', color='b')
# Zaznaczenie maksimum na obrazie wejściowym
output_img = cv2.cvtColor(search_img, cv2.COLOR_GRAY2BGR)
cv2.circle(output_img, (max_x, max_y), 10, (255, 0, 0))


hough_space = np.uint8(hough_space * 255.0 / hough_space.max())

# Wyświetlenie wyników
#cv2.imshow('Pattern', pattern)
cv2.imshow('Hough Space', hough_space)
cv2.imshow('Result', output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

