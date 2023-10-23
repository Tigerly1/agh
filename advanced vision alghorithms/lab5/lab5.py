# Zadanie 5.1 Zaimplementuj metod˛e blokow ˛a do wyznaczania przepływu optycznego.
# 1. Utwórz nowy skrypt w j˛ezyku Python. Wczytaj obrazy I.jpg i J.jpg. B˛ed ˛a to dwie
# ramki z sekwencji, oznaczone jako I i J (odpowiednio wczesniejsza i pó´zniejsza ´
# ramka). Opcjonalnie mozesz zmniejszy ˙ c oba obrazy na czas testów, np. czterokrotnie – ´
# obliczenia b˛ed ˛a si˛e wykonywały szybciej. Wykonaj konwersj˛e obrazów do odcieni szarosci – ´ cvtColor. Wyswietl zadane obrazy – przydatne mo ´ ze by ˙ c wy ´ swietlanie przy ´
# uzyciu ˙ namedWindow oraz imshow. Zwizualizuj róznic˛e, wykorzystuj ˛ac polecenie: ˙
# absdiff.
# 2. Wykorzystuj ˛ac ponizsze wskazówki, zaimplementuj metod˛e blokow ˛a wyznaczania ˙
# przepływu optycznego. Przyjmij nast˛epuj ˛ace załozenia – porównujemy fragmenty ˙
# obrazu o rozmiarze 7×7. Wykorzystywane dalej oznaczenia oraz wartosci dla okna ´
# 41
# 7×7:
# • W2 = 3 – wartos´c całkowita z połowy rozmiaru okna, ´
# • dX = dY = 3 – rozmiar przeszukiwania (maksymalnego przesuwania okna) w obu
# kierunkach.
# 3. Implementacj˛e nalezy zacz ˛a ˙ c od dwóch p˛etli ´ for po obrazie. Wykorzystaj parametr
# W2 do warunku uwzgl˛edniaj ˛acego przypadek brzegowy – zakładamy, ze na brzegu nie ˙
# wyliczamy przepływu optycznego.
# 4. Wewn ˛atrz p˛etli wycinamy fragment ramki I. Dla przypomnienia, niezb˛edna składnia
# to: IO = np.float32(I[j-W2:j+W2+1,i-W2:i+W2+1]).
# Uwaga!
# W rozwazaniach przyjmujemy, ˙ ze˙ j – indeks p˛etli zewn˛etrznej (po wierszach), i –
# indeks p˛etli wewn˛etrznej (po kolumnach).
# Prosz˛e zwrócic uwag˛e na składnik „+1” – w Pythonie górny zakres to warto ´ s´c o 1 wi˛ek- ´
# sza od najwi˛ekszej przyjmowanej wartosci – inaczej ni ´ z w Matlabie. Konwersja na typ ˙
# zmiennoprzecinkowy potrzebna jest do dalszych obliczen.´
# 5. Nast˛epnie realizujemy kolejne dwie p˛etle for – przeszukiwanie otoczenia piksela
# J(j,i). Maj ˛a one zakres od −dX do dX i −dY do dY. Prosz˛e nie zapomniec´
# o „+1”. Wewn ˛atrz p˛etli nalezy sprawdzi ˙ c, czy współrz˛edne aktualnego kontekstu ´
# (tj. jego srodek) mieszcz ˛a si˛e w dopuszczalnym zakresie. Alternatywnie mo ´ zna te ˙ z˙
# zmodyfikowac zakres zewn˛etrznych p˛etli – tak aby wykluczy ´ c dost˛ep do pikseli spoza ´
# dopuszczalnego zakresu. W tym przypadku dla nieco szerszego brzegu przepływ nie
# zostanie okreslony, jednak nie ma to istotnego znaczenia praktycznego. ´
# 6. Wycinamy otoczenie JO, dokonujemy konwersji na float32 a nast˛epnie obliczamy
# „odległos´c” mi˛edzy wycinkami ´ IO a JO. Mozna to wykona ˙ c instrukcj ˛a: ´ np.sum(np.
# sqrt((np.square(JO-IO)))). Sposród wszystkich wycinków z ´ JO dla danego
# IO nalezy znale´z ˙ c najmniejsz ˛a „odległo ´ s´c” – lokalizacj˛e „najbardziej podobnego” do ´
# IO fragmentu na obrazie J.
# 7. Uzyskane współrz˛edne znalezionych minimów nalezy zapisa ˙ c w dwóch macierzach ´
# (przykładowo u i v) – nalezy je wcze ˙ sniej (przed główn ˛a p˛etl ˛a) utworzy ´ c i zainicjowa ´ c´
# zerami (funkcja np.zeros), wymiary takie jak dla obrazu I.
# 8. Wyznaczone pole przepływu optycznego mozna zwizualizowa ˙ c na dwa sposoby – ´
# poprzez wektory (strzałki) – funkcja plt.quiver z biblioteki matplotlib lub kolory –
# zrealizujemy drugie z tych podejs´c. Pomysł opiera si˛e na okre ´ sleniu k ˛ata i długo ´ sci ´
# wektora wyznaczonego przez dwie składowe przepływu optycznego u i v dla kazdego ˙
# piksela. Otrzymamy wówczas reprezentacj˛e danych podobn ˛a jak w przestrzeni barw
# HSV. Po wyswietleniu kolor oznacza kierunek, w którym przemieszczaj ˛a si˛e piksele, ´
# natomiast jego nasycenie informuje o wzgl˛ednej szybkosci ruchu pikseli – przeanalizuj ´
# koło kolorów na rys. 5.2.
# Dokonaj konwersji wyznaczonego przepływu do układu współrz˛ednych biegunowych
# – cartToPolar. Utwórz zmienn ˛a na obraz w przestrzeni HSV o wymiarach wejsciowego obrazu, 3 kanałach i typie ´ uint8. Pierwszy kanał to składowa H, równa
# angle ∗ 90/np.pi (w Pythonie zakres od 0 do 180). Drugi kanał to składowa S – dokonaj normalizacji (normalize) długosci wektora do przedziału 0-255. Trzeci kanał to ´
# składowa V, ustaw j ˛a na 255.
# Uwaga!
# Aby brak ruchu oznaczony był kolorem czarnym, a nie białym, nalezy zamieni ˙ c´
# kanały S i V
# Na koniec wykonaj konwersj˛e obrazu do przestrzeni RGB i wyswietl go. ´
# Prosz˛e przeanalizowac uzyskane wyniki oraz poeksperymentowa ´ c z rozmiarem obrazu ´
# wejsciowego oraz z parametrami ´ W2 oraz dX oraz dY (np. obraz dwukrotnie zmniejszony, parametry równe 5). Czy dla obrazu wejsciowego w oryginalnych rozmiarach ´
# takie parametry pozwalaj ˛a poprawnie wyznaczyc przepływ optyczny? Jak du ´ ze okno ˙
# jest potrzebne, aby otrzymac podobne wyniki? ´
# Sprawd´z działanie metody dla pary obrazów z innej sekwencji – cm1.png i cm2.png.
# Ground truth dla tej pary obrazów, uzywane do ewaluacji algorytmów wyznaczania ˙
# przepływu optycznego, zamieszczono na rys. 5.3.
# Uwaga!
# Metoda blokowa z uwagi na sw ˛a prostot˛e jest dos´c niedokładna – wyliczone przesu- ´
# ni˛ecie w poziomie lub w pionie jest całkowitoliczbowe. W innych metodach (np.
# HS czy LK) przepływ optyczny jest przewaznie niecałkowity. ˙
# ■


import cv2
import numpy as np
import matplotlib.pyplot as plt
 
I = cv2.imread('I.jpg', cv2.IMREAD_GRAYSCALE)
J = cv2.imread('J.jpg', cv2.IMREAD_GRAYSCALE)

new_shape = (I.shape[1] // 4, J.shape[0] // 4)


I = cv2.resize(I, new_shape, interpolation = cv2.INTER_LINEAR )
J = cv2.resize(J, new_shape, interpolation = cv2.INTER_LINEAR )


W2 = 3
dX = 3

u = np.zeros(I.shape)
v = np.zeros(I.shape)

for j in range(W2, I.shape[0]-W2):
    for i in range(W2, I.shape[1]-W2):
        IO = np.float32(I[j-W2:j+W2+1, i-W2:i+W2+1])
        min_dist = np.inf
        for dY in range(-5, 5+1):
            for dX in range(-5, 5+1):
                if j+dY >= W2 and j+dY < I.shape[0]-W2 and i+dX >= W2 and i+dX < I.shape[1]-W2:
                    JO = np.float32(J[j+dY-W2:j+dY+W2+1, i+dX-W2:i+dX+W2+1])
                    dist = np.sum(np.sqrt((np.square(JO-IO))))
                    if dist < min_dist:
                        min_dist = dist
                        u[j, i] = dX
                        v[j, i] = dY

plt.imshow(I, 'gray')
plt.show()
plt.imshow(J, 'gray')
plt.show()

plt.imshow(u, 'gray')
plt.show()
plt.imshow(v, 'gray')
plt.show()

u = cv2.normalize(u, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
v = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

plt.imshow(u, 'gray')
plt.show()

plt.imshow(v, 'gray')
plt.show()

mag, ang = cv2.cartToPolar(u, v, angleInDegrees=True)

plt.imshow(mag, 'gray')
plt.show()

plt.imshow(ang, 'gray')
plt.show()

hsv = np.zeros((I.shape[0], I.shape[1], 3), dtype=np.uint8)
hsv[..., 0] = ang/2
hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
hsv[..., 2] = 255
 
rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
 
plt.imshow(rgb)
plt.show()
 
cv2.imshow('rgb', rgb)

cv2.waitKey(0)

cv2.destroyAllWindows()
 


#  Zadanie 5.2 Zaimplementuj wieloskalow ˛a wersj˛e metody blokowej do wyznaczania
# przepływu optycznego.
# 1. Na wst˛epie prosz˛e stworzyc kopi˛e dotychczas stworzonego algorytmu w ramach zada- ´
# nia 1. Nast˛epnie fragment algorytmu dotycz ˛acy wyliczania OF zamieniamy na funkcj˛e
# dla wybranej skali.
# Funkcja powinna miec nast˛epuj ˛ac ˛a posta ´ c:´
# def of(I_org, I, J, W2=3, dY=3, dX=3):
# Poszczególne parametry to I_org i J – obrazy wejsciowe, ´ I – obraz po modyfikacji,
# W2, dX, dY – parametry metody (identyczne jak w 5.3). Tak naprawd˛e przesyłanie
# I_org do funkcji nie jest potrzebne – sugerowane rozwi ˛azanie ma na celu obliczenie
# absdiff mi˛edzy obrazami i wyswietlenie wyników oraz obrazów ´ I_org, I oraz J
# dla lepszego zrozumienia działania metody. W rzeczywistosci obliczenia prowadzone ´
# s ˛a dla I i J.
# 2. Przydatne b˛edzie takze zamienienie fragmentu algorytmu dotycz ˛acego wizualizacji ˙
# przepływu optycznego na funkcj˛e. Funkcja ta powinna byc postaci: ´
# def vis_flow(u, v, YX, name):
# Poszczególne parametry to u i v – składowe przepływu optycznego, YX – wymiary
# obrazu, name – wyswietlana nazwa okna, np. ´ ’of scale 2’.
# 3. Po napisaniu funkcji dobrze jest przetestowac jej działanie dla przypadku jednej skali ´
# (L0) – wynik powinien byc identyczny jak otrzymywany wcze ´ sniej. ´
# 4. Po utworzeniu dwóch funkcji na podstawie napisanego juz algorytmu, przechodzimy do ˙
# napisania funkcji do generowania piramidy obrazów. Mozna wykorzysta ˙ c nast˛epuj ˛ac ˛a ´
# funkcj˛e:
# def pyramid(im, max_scale):
# images=[im]
# for k in range(1, max_scale):
# images.append(cv2.resize(images[k-1], (0,0), fx=0.5, fy=0.5))
# return images
# W tym przypadku zakładamy, ze pomniejszenia rozdzielczo ˙ sci zawsze b˛ed ˛a dwukrotne. ´
# Czyli obraz wejsciowy pomniejszamy 2-krotnie, 4-krotnie, 8-krotnie itd. Na potrzeby ´
# eksperymentu zakładamy, ze ograniczamy si˛e do 3 skal. Piramid˛e nale ˙ zy wygenerowa ˙ c´
# dla kazdego z obrazów wej ˙ sciowych. ´
# 5. Przetwarzanie zaczynamy od skali najmniejszej, zatem do zmiennej I przypisujemy
# obraz z piramidy w najmniejszej skali: I = IP[-1], gdzie IP to wygenerowana
# piramida dla pierwszego obrazu. Prosz˛e zwrócic uwag˛e na składni˛e ´ [-1] – dost˛ep do
# ostatniego elementu.
# 6. Kluczowy komponent algorytmu to p˛etla po skalach – prosz˛e zastanowic si˛e nad indek- ´
# 45
# sem pocz ˛atkowym i koncowym. Zaczynamy od wyliczenia przepływu i robimy kopi˛e ´
# pierwszego obrazu I_new. Nast˛epny krok to modyfikacja tej kopii zgodnie z przepływem. T˛e operacj˛e wykonujemy dla wszystkich skal oprócz najwi˛ekszej (czyli obrazu
# o wejsciowych wymiarach). Mo ´ zna to zrealizowa ˙ c przy wykorzystaniu dwóch p˛etli ´
# for z ew. zabezpieczeniem przed wyjsciem poza zakres. Poszczególnym pikselom ´
# z I_new przypisujemy odpowiednie piksele z I, zgodnie z wyliczonym przepływem.
# Warto sprawdzic, czy obraz pierwszy po modyfikacji jest zbli ´ zony do obrazu drugiego – ˙
# jesli tak jest, to operacja została zrealizowana poprawnie. Na koniec musimy przygoto- ´
# wac obraz do kolejnej (wi˛ekszej) skali. W tym celu zwi˛ekszamy obraz ´ I_new dwukrotnie i przypisujemy go do I. Zwi˛ekszenie: I = cv2.resize(I_new, (0,0),
# fx=2, fy=2, interpolation=cv2.INTER_LINEAR).
# 7. Ostatni element algorytmu to wyliczenie całkowitego przepływu i jego wizualizacja.
# W tym celu nalezy przygotowa ˙ c pust ˛a tablic˛e 2D dla ka ´ zdej ze składowych ˙ u i v
# o wymiarach wejsciowego obrazu. Nast˛epnie w p˛etli po skalach dodajemy do siebie ´
# przepływy z róznych skal. ˙
# Wskazówka
# Krótki przykład dla wyjasnienia. Mamy piksel, którego przemieszczenie wynosi 9 ´
# (ground truth), jednak przeszukiwanie mamy ograniczone do 5 pikseli w kazdym ˙
# kierunku. Obraz zmniejszamy dwa razy, teraz przemieszczenie piksela powinno
# wynosic 4.5. Znajdujemy najbardziej pasuj ˛acy piksel w mniejszej skali – otrzymu- ´
# jemy przesuni˛ecie równe 5, zatem zgodnie z tym wynikiem modyfikujemy obraz
# (przesuwamy piksel o 5 w odpowiednim kierunku) i zwi˛ekszamy go dwa razy –
# teraz (w wi˛ekszej skali) analizowany piksel jest przesuni˛ety o 10. Liczymy ponownie przepływ, tym razem w wi˛ekszej skali, i otrzymujemy przesuni˛ecie równe
# -1 (przepływ rezydualny). Nast˛epnie sumujemy wyniki z obu skal i otrzymujemy
# przesuni˛ecie równe 9.
# Wnioski. Aby otrzymac poprawny wynik ko ´ ncowy, nale ´ zy przepływy ˙ u i v z kazdej ˙
# skali zwi˛ekszyc na dwa sposoby – wymiary tych „obrazów” musz ˛a by ´ c takie, jak ´
# dla wejsciowego ´ I, a wartosci przepływów musz ˛a by ´ c zwi˛ekszone 2, 4, 8 itd. razy, ´
# w zalezno ˙ sci od skali. Przepływ zsumowany z ró ´ znych skal nale ˙ zy zwizualizowa ˙ c przy ´
# pomocy funkcji vis_flow.
# 8. Porównaj działanie metody ze skalami i bez. Przeanalizuj wyniki dla wi˛ekszych
# rozmiarów okien w jednej skali oraz dla mniejszych rozmiarów okien z wieloma
# skalami, zwróc uwag˛e na czas wykonania. Czy stosuj ˛ac metod˛e wieloskalow ˛a (np. 3 ´
# skale), udało si˛e uzyskac poprawny przepływ dla niewielkiego okna (np. ´ dX = dY = 3)
# dla obrazu o oryginalnym rozmiarze (czyli dla wi˛ekszego przemieszczenia pikseli)?
# Ponownie sprawd´z działanie metody dla pary obrazów cm1.png i cm2.png.
# Uwaga!
# Metoda wieloskalowa w teorii działa bardzo dobrze – w praktyce dopasowanie
# pikseli nie zawsze jest dokładne, a skalowanie w dół i w gór˛e zawsze powoduje
# utrat˛e pewnych informacji, przez co nawet drobne bł˛edy propagowane s ˛a na kolejne
# skale i koncowy wynik mo ´ ze by ˙ c cz˛e ´ sciowo nieprawidłowy b ˛ad´z zaszumiony. ´
# Z tego powodu w praktyce uzywa si˛e 2, 3, maksymalnie 4 skale (ale i tak wyniki ˙
# rózni ˛a si˛e nieco od „oczekiwanych”). ˙

import cv2
import numpy as np
import matplotlib.pyplot as plt

def of(I, J, W2=5, dY=5, dX=5):   
    u = np.zeros(I.shape)
    v = np.zeros(I.shape)
    for j in range(W2, I.shape[0]-W2):
        for i in range(W2, I.shape[1]-W2):
            IO = np.float32(I[j-W2:j+W2+1, i-W2:i+W2+1])
            min_dist = np.inf
            for dY in range(-W2, W2+1):
                for dX in range(-W2, W2+1):
                    if j+dY >= W2 and j+dY < I.shape[0]-W2 and i+dX >= W2 and i+dX < I.shape[1]-W2:
                        JO = np.float32(J[j+dY-W2:j+dY+W2+1, i+dX-W2:i+dX+W2+1])
                        dist = np.sum((JO-IO)**2)
                        if dist < min_dist:
                            min_dist = dist
                            u[j, i] = dX
                            v[j, i] = dY
    return u, v

def vis_flow(u, v, YX, name):
    flow_image = np.sqrt(u**2 + v**2)
    plt.imshow(flow_image)
    plt.title(name)
    plt.show()

def pyramid(im, max_scale):
    images=[im]
    for k in range(1, max_scale):
        images.append(cv2.resize(images[k-1], (0,0), fx=0.5, fy=0.5))
    return images

# Loading your images
I_org = cv2.imread('I.jpg', 0)
J = cv2.imread('J.jpg', 0)

# Generating image pyramid
max_scale = 3
IP = pyramid(I_org, max_scale)
JP = pyramid(J, max_scale)

# Prepare arrays for u, v
YX = I_org.shape
u_total = np.zeros(YX)
v_total = np.zeros(YX)

# Loop over scales
for scale in reversed(range(max_scale)):
    I = IP[scale]
    J = JP[scale]

    # Compute optical flow for this scale
    u, v = of(I, J)

    # Resize the flow to original image size
    u_resized = cv2.resize(u, (YX[1], YX[0]), interpolation=cv2.INTER_LINEAR) * (2 ** scale)
    v_resized = cv2.resize(v, (YX[1], YX[0]), interpolation=cv2.INTER_LINEAR) * (2 ** scale)

    u_total += u_resized
    v_total += v_resized

    vis_flow(u, v, YX, f'Optical Flow scale {scale}')

# Final optical flow visualization
vis_flow(u_total, v_total, YX, 'Total Optical Flow')