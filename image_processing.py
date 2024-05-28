import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import math
from collections import Counter
from pylab import savefig
import cv2
import os
import csv

def grayscale():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    r = img_arr[:, :, 0]
    g = img_arr[:, :, 1]
    b = img_arr[:, :, 2]
    new_arr = r.astype(int) + g.astype(int) + b.astype(int)
    new_arr = (new_arr/3).astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")

def binary(img_path):
    img = Image.open(img_path)
    img_arr = np.asarray(img)
    r = img_arr[:, :, 0]
    g = img_arr[:, :, 1]
    b = img_arr[:, :, 2]
    new_arr = (r + g + b) // 3  
    new_arr = (new_arr >= 128) * 255 
    new_img = Image.fromarray(new_arr.astype('uint8'), 'L')  
    new_img.save("static/img/img_now_binary.jpg")


from PIL import Image
import numpy as np


def is_grey_scale(img_path):
    im = Image.open(img_path).convert('RGB')
    w, h = im.size
    for i in range(w):
        for j in range(h):
            r, g, b = im.getpixel((i, j))
            if r != g != b:
                return False
    return True

def binary(img_path):
    img = Image.open(img_path)
    img_arr = np.asarray(img)
    
    # Assuming img_arr has shape (height, width)
    # Check the shape of img_arr
    print("Shape of img_arr:", img_arr.shape)
    
    # Convert the image to grayscale
    img_gray = img.convert('L')
    img_arr_gray = np.asarray(img_gray)
    
    # Compute the binary image
    threshold = 128
    binary_arr = (img_arr_gray >= threshold) * 255
    
    # Convert the binary array to an image
    binary_img = Image.fromarray(binary_arr.astype('uint8'), 'L')
    
    # Save the binary image
    binary_img.save("static/img/img_now_binary.jpg")



def is_binary(img_path, threshold=128):
    im = Image.open(img_path).convert('L')  # Convert image to grayscale
    w, h = im.size
    for i in range(w):
        for j in range(h):
            pixel_value = im.getpixel((i, j))
            if pixel_value != 0 and pixel_value != 255: 
                return False
    return True





def zoomin():
    img = Image.open("static/img/img_now.jpg")
    img = img.convert("RGB")
    img_arr = np.asarray(img)
    new_size = ((img_arr.shape[0] * 2),
                (img_arr.shape[1] * 2), img_arr.shape[2])
    new_arr = np.full(new_size, 255)
    new_arr.setflags(write=1)

    r = img_arr[:, :, 0]
    g = img_arr[:, :, 1]
    b = img_arr[:, :, 2]

    new_r = []
    new_g = []
    new_b = []

    for row in range(len(r)):
        temp_r = []
        temp_g = []
        temp_b = []
        for i in r[row]:
            temp_r.extend([i, i])
        for j in g[row]:
            temp_g.extend([j, j])
        for k in b[row]:
            temp_b.extend([k, k])
        for _ in (0, 1):
            new_r.append(temp_r)
            new_g.append(temp_g)
            new_b.append(temp_b)

    for i in range(len(new_arr)):
        for j in range(len(new_arr[i])):
            new_arr[i, j, 0] = new_r[i][j]
            new_arr[i, j, 1] = new_g[i][j]
            new_arr[i, j, 2] = new_b[i][j]

    new_arr = new_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def zoomout():
    img = Image.open("static/img/img_now.jpg")
    img = img.convert("RGB")
    x, y = img.size
    new_arr = Image.new("RGB", (int(x / 2), int(y / 2)))
    r = [0, 0, 0, 0]
    g = [0, 0, 0, 0]
    b = [0, 0, 0, 0]

    for i in range(0, int(x/2)):
        for j in range(0, int(y/2)):
            r[0], g[0], b[0] = img.getpixel((2 * i, 2 * j))
            r[1], g[1], b[1] = img.getpixel((2 * i + 1, 2 * j))
            r[2], g[2], b[2] = img.getpixel((2 * i, 2 * j + 1))
            r[3], g[3], b[3] = img.getpixel((2 * i + 1, 2 * j + 1))
            new_arr.putpixel((int(i), int(j)), (int((r[0] + r[1] + r[2] + r[3]) / 4), int(
                (g[0] + g[1] + g[2] + g[3]) / 4), int((b[0] + b[1] + b[2] + b[3]) / 4)))
    new_arr = np.uint8(new_arr)
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def move_left():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    r = np.pad(r, ((0, 0), (0, 50)), 'constant')[:, 50:]
    g = np.pad(g, ((0, 0), (0, 50)), 'constant')[:, 50:]
    b = np.pad(b, ((0, 0), (0, 50)), 'constant')[:, 50:]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def move_right():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    r = np.pad(r, ((0, 0), (50, 0)), 'constant')[:, :-50]
    g = np.pad(g, ((0, 0), (50, 0)), 'constant')[:, :-50]
    b = np.pad(b, ((0, 0), (50, 0)), 'constant')[:, :-50]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def move_up():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    r = np.pad(r, ((0, 50), (0, 0)), 'constant')[50:, :]
    g = np.pad(g, ((0, 50), (0, 0)), 'constant')[50:, :]
    b = np.pad(b, ((0, 50), (0, 0)), 'constant')[50:, :]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def move_down():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
    r = np.pad(r, ((50, 0), (0, 0)), 'constant')[0:-50, :]
    g = np.pad(g, ((50, 0), (0, 0)), 'constant')[0:-50, :]
    b = np.pad(b, ((50, 0), (0, 0)), 'constant')[0:-50, :]
    new_arr = np.dstack((r, g, b))
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def brightness_addition():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img).astype('uint16')
    img_arr = img_arr+100
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def brightness_substraction():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img).astype('int16')
    img_arr = img_arr-100
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def brightness_multiplication():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    img_arr = img_arr*1.25
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def brightness_division():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.asarray(img)
    img_arr = img_arr/1.25
    img_arr = np.clip(img_arr, 0, 255)
    new_arr = img_arr.astype('uint8')
    new_img = Image.fromarray(new_arr)
    new_img = new_img.convert("RGB")
    new_img.save("static/img/img_now.jpg")


def convolution(img, kernel):
    h_img, w_img, _ = img.shape
    out = np.zeros((h_img-2, w_img-2), dtype=np.float)
    new_img = np.zeros((h_img-2, w_img-2, 3))
    if np.array_equal((img[:, :, 1], img[:, :, 0]), img[:, :, 2]) == True:
        array = img[:, :, 0]
        for h in range(h_img-2):
            for w in range(w_img-2):
                S = np.multiply(array[h:h+3, w:w+3], kernel)
                out[h, w] = np.sum(S)
        out_ = np.clip(out, 0, 255)
        for channel in range(3):
            new_img[:, :, channel] = out_
    else:
        for channel in range(3):
            array = img[:, :, channel]
            for h in range(h_img-2):
                for w in range(w_img-2):
                    S = np.multiply(array[h:h+3, w:w+3], kernel)
                    out[h, w] = np.sum(S)
            out_ = np.clip(out, 0, 255)
            new_img[:, :, channel] = out_
    new_img = np.uint8(new_img)
    return new_img


def edge_detection():
    img = Image.open("static/img/img_now.jpg").convert('L')
    img_arr = np.array(img)
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    new_arr = cv2.filter2D(img_arr, -1, kernel)
    new_img = Image.fromarray(new_arr)
    new_img.save("static/img/img_now.jpg")


def blur():
    import PIL
    try:
        img = Image.open("static/img/img_now.jpg")
        img_arr = np.array(img)
        blurred_img = cv2.GaussianBlur(img_arr, (5, 5), 0)
        new_img = Image.fromarray(blurred_img).convert('RGB')
        new_img.save("static/img/img_now.jpg")
    except PIL.UnidentifiedImageError as e:
        print("Error:", e)


def sharpening():
    img = Image.open("static/img/img_now.jpg")
    img_arr = np.array(img)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)  # Kernel sharpening
    sharpened_img = cv2.filter2D(img_arr, -1, kernel)
    sharpened_img = np.clip(sharpened_img, 0, 255).astype(np.uint8)  # Clip hasil agar tetap dalam rentang 0-255
    new_img = Image.fromarray(sharpened_img).convert("RGB")  # Konversi ke mode warna RGB
    new_img.save("static/img/img_now.jpg")


def histogram_rgb():
    img_path = "static/img/img_now.jpg"
    img = Image.open(img_path)
    img_arr = np.asarray(img)
    if is_grey_scale(img_path):
        g = img_arr[:, :, 0].flatten()
        data_g = Counter(g)
        plt.bar(list(data_g.keys()), data_g.values(), color='black')
        plt.savefig(f'static/img/grey_histogram.jpg', dpi=300)
        plt.clf()
    else:
        r = img_arr[:, :, 0].flatten()
        g = img_arr[:, :, 1].flatten()
        b = img_arr[:, :, 2].flatten()
        data_r = Counter(r)
        data_g = Counter(g)
        data_b = Counter(b)
        data_rgb = [data_r, data_g, data_b]
        warna = ['red', 'green', 'blue']
        data_hist = list(zip(warna, data_rgb))
        for data in data_hist:
            plt.bar(list(data[1].keys()), data[1].values(), color=f'{data[0]}')
            plt.savefig(f'static/img/{data[0]}_histogram.jpg', dpi=300)
            plt.clf()


def df(img):  # to make a histogram (count distribution frequency)
    values = [0]*256
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            values[img[i, j]] += 1
    return values


def cdf(hist):  # cumulative distribution frequency
    cdf = [0] * len(hist)  # len(hist) is 256
    cdf[0] = hist[0]
    for i in range(1, len(hist)):
        cdf[i] = cdf[i-1]+hist[i]
    # Now we normalize the histogram
    # What your function h was doing before
    cdf = [ele*255/cdf[-1] for ele in cdf]
    return cdf

def histogram_equalizer():
    img = cv2.imread('static\img\img_now.jpg', 0)
    my_cdf = cdf(df(img))
    # use linear interpolation of cdf to find new pixel values. Scipy alternative exists
    image_equalized = np.interp(img, range(0, 256), my_cdf)
    # Convert to RGB mode before saving
    new_img = Image.fromarray(image_equalized).convert('RGB')
    new_img.save('static/img/img_now.jpg')


def threshold(lower_thres, upper_thres):
    # Periksa apakah lower_thres dan upper_thres merupakan bilangan bulat positif
    if isinstance(lower_thres, int) and isinstance(upper_thres, int) and lower_thres >= 0 and upper_thres >= 0:
        img = Image.open("static/img/img_now.jpg").convert('L')
        img_arr = np.array(img)
        _, segmented_img = cv2.threshold(img_arr, lower_thres, upper_thres, cv2.THRESH_BINARY)
        new_img = Image.fromarray(segmented_img)
        new_img.save("static/img/img_now.jpg")
    else:
        print("Error: Lower and upper thresholds must be positive integers.")

def erosi():
    img = Image.open("static/img/img_now.jpg")
    kernel = np.ones((5, 5), np.uint8)
    erosi_img = cv2.erode(np.array(img), kernel, iterations=1)
    img = Image.fromarray(erosi_img )
    img.save("static/img/img_now.jpg")


def dilasi():
    img = Image.open("static/img/img_now.jpg")
    kernel = np.ones((5, 5), np.uint8)
    dilasi_img = cv2.dilate(np.array(img), kernel, iterations=1)
    new_img = Image.fromarray(dilasi_img )
    new_img.save("static/img/img_now.jpg")


def opening():
    img = Image.open("static/img/img_now.jpg")
    kernel = np.ones((5, 5), np.uint8)
    opening_img = cv2.morphologyEx(np.array(img), cv2.MORPH_OPEN, kernel)
    new_img = Image.fromarray(opening_img)
    new_img.save("static/img/img_now.jpg")

def closing():
    img = Image.open("static/img/img_now.jpg")
    kernel = np.ones((5, 5), np.uint8)
    closing_img = cv2.morphologyEx(np.array(img), cv2.MORPH_CLOSE, kernel)
    new_img = Image.fromarray(closing_img )
    new_img.save("static/img/img_now.jpg")

def count1():
    img_gray = cv2.imread('static/img/img_now.jpg', cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_objects = len(contours)
    return num_objects

def count2():
    img = Image.open("static/img/img_now.jpg").convert('L')
    img_np = np.array(img)
    img_blur = cv2.GaussianBlur(img_np, (5, 5), 0)
    _, thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=1)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(
        dist_transform, 0.4*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    img_color = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)

    num_objects = len(np.unique(markers))
    img_color[markers == -1] = [255, 0, 0]
    return num_objects

def count3():
    img = cv2.imread('static/img/img_now.jpg', 0)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_objects = len(contours)
    return num_objects

def create_digit_images():
    # Font yang akan digunakan untuk digit
    font = ImageFont.truetype("arial.ttf", 48)
    
    # Ukuran citra (width x height)
    image_size = (64, 64)
    
    for digit in range(10):
        # Membuat citra kosong
        image = Image.new("L", image_size, color=255)
        
        # Membuat objek Draw
        draw = ImageDraw.Draw(image)
        
        # Menentukan posisi untuk menulis digit agar berada di tengah
        text_width, text_height = draw.textsize(str(digit), font=font)
        position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2)
        
        # Menulis digit ke dalam citra
        draw.text(position, str(digit), fill=0, font=font)
        
        # Menyimpan citra
        image.save(f"static/digits/{digit}.png")

def extract_contours():
    # Membuat direktori untuk menyimpan hasil ekstraksi kontur
    if not os.path.exists("static/contours/"):
        os.makedirs("static/contours/")
    
    # Loop untuk setiap citra angka digit
    for digit in range(10):
        # Memuat citra
        image_path = f"static/digits/{digit}.png"
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Melakukan Canny edge detection
        edges = cv2.Canny(img, 100, 200)
        
        # Menyimpan hasil ekstraksi kontur
        cv2.imwrite(f"static/contours/contour_digit_{digit}.png", edges)

def freeman_chain_code():
    # Membuat direktori untuk menyimpan hasil FCC
    if not os.path.exists("static/fcc/"):
        os.makedirs("static/fcc/")
    
    # Loop untuk setiap citra kontur
    for digit in range(10):
        # Memuat citra kontur
        contour_path = f"static/contours/contour_digit_{digit}.png"
        contour_img = cv2.imread(contour_path, cv2.IMREAD_GRAYSCALE)
        
        # Mencari kontur menggunakan OpenCV
        contours, _ = cv2.findContours(contour_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # Memperoleh kontur dengan area terbesar
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Menghitung FCC dari kontur
        fcc = ""
        for point in largest_contour:
            x, y = point[0]
            fcc += str(x) + "," + str(y) + " "
        
        # Menyimpan hasil FCC
        with open(f"static/fcc/fcc_digit_{digit}.txt", "w") as file:
            file.write(fcc)

def zhang_suen_thinning():
    if not os.path.exists("static/thinning/"):
        os.makedirs("static/thinning/")
    
    for digit in range(10):
        contour_path = f"static/contours/contour_digit_{digit}.png"
        contour_img = cv2.imread(contour_path, cv2.IMREAD_GRAYSCALE)
        thinned_img = contour_img.copy()
        
        changed = True
        while changed:
            changed = False
            thinned_img, changed = apply_zhang_suen_iteration(thinned_img)
        
        save_thinned_image(thinned_img, digit)


def apply_zhang_suen_iteration(img: np.ndarray) -> tuple[np.ndarray, bool]:
    temp_img = img.copy()
    changed = False
    
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if is_pixel_to_be_deleted(img, i, j):
                temp_img[i, j] = 0
                changed = True
    
    return temp_img, changed


def is_pixel_to_be_deleted(img: np.ndarray, i: int, j: int) -> bool:
    # Periksa apakah piksel saat ini adalah piksel putih (255)
    if img[i, j] == 255:
        # Hitung jumlah piksel tetangga yang berwarna hitam (0)
        neighbors_sum = img[i-1:i+2, j-1:j+2].sum()
        # Periksa apakah jumlah tetangga adalah di antara 2 dan 6
        if 2 <= neighbors_sum <= 6:
            # Periksa pola piksel sesuai dengan aturan algoritma Zhang-Suen
            if ((img[i-1, j] == 0 or img[i, j+1] == 0) and img[i+1, j] == 0) or ((img[i-1, j] == 0 or img[i, j-1] == 0) and img[i+1, j] == 0):
                return True
    return False


def save_thinned_image(img: np.ndarray, digit: int) -> None:
    cv2.imwrite(f"static/thinning/thinned_digit_{digit}.png", img)

def save_to_csv():
    # Membuka file CSV untuk disimpan
    with open('knowledge_base.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Digit', 'FCC', 'Contour', 'Thinning'])
        
        # Loop untuk setiap digit
        for digit in range(10):
            # Simpan data ke dalam baris CSV
            fcc_path = f"static/fcc/fcc_digit_{digit}.txt"
            contour_path = f"static/contours/contour_digit_{digit}.png"
            thinning_path = f"static/thinning/thinned_digit_{digit}.png"
            writer.writerow([digit, fcc_path, contour_path, thinning_path])

def identify_uploaded_image(image_path):
    # Implementasi fungsi untuk mengidentifikasi angka berdasarkan FCC
    
    # Memuat citra yang diunggah
    uploaded_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Melakukan Canny edge detection pada citra yang diunggah
    uploaded_edges = cv2.Canny(uploaded_img, 100, 200)
    
    # Menyimpan citra kontur yang diunggah
    cv2.imwrite("static/contours/uploaded_contour.png", uploaded_edges)
    
    # Menghitung FCC dari kontur yang diunggah
    uploaded_fcc = ""
    contours, _ = cv2.findContours(uploaded_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest_contour = max(contours, key=cv2.contourArea)
    for point in largest_contour:
        x, y = point[0]
        uploaded_fcc += str(x) + "," + str(y) + " "
    
    # Menyimpan hasil FCC dari kontur yang diunggah
    with open("static/fcc/uploaded_fcc.txt", "w") as file:
        file.write(uploaded_fcc)
    
    # Membandingkan FCC dari citra yang diunggah dengan FCC dari setiap digit
    min_distance = float('inf')
    identified_digit = None
    for digit in range(10):
        # Memuat FCC dari digit
        with open(f"static/fcc/fcc_digit_{digit}.txt", "r") as file:
            digit_fcc = file.read()
        
        # Menghitung jarak antara FCC dari citra yang diunggah dan FCC dari digit
        distance = calculate_fcc_distance(uploaded_fcc, digit_fcc)
        
        # Memperbarui digit yang teridentifikasi jika jaraknya lebih kecil
        if distance < min_distance:
            min_distance = distance
            identified_digit = digit
    
    return identified_digit 

def calculate_fcc_distance(fcc1, fcc2):
    # Menghitung jarak antara dua rangkaian Freeman Chain Code
    fcc1_points = fcc1.split()
    fcc2_points = fcc2.split()
    distance = 0
    for p1, p2 in zip(fcc1_points, fcc2_points):
        x1, y1 = map(int, p1.split(","))
        x2, y2 = map(int, p2.split(","))
        distance += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    distance = sum((int(a) - int(b)) ** 2 for a, b in zip(fcc1.split(), fcc2.split()))
    return distance

def generate_chaincode(thickness):
    chain = []
    for t in thickness:
        if t == 1:
            chain.append(1)
        else:
            chain.extend([0, 1])
    return chain

def process_image1(file_path):
    try:
        # Membaca gambar baru
        new_image = cv2.imread(file_path)
        original_image = new_image.copy()  # Copy the original image

        # Mengubah gambar baru menjadi skala abu-abu
        new_gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

        # Terapkan deteksi tepi menggunakan Canny edge detector
        new_edges = cv2.Canny(new_gray, 50, 150)

        # Temukan kontur menggunakan fungsi findContours
        new_contours, _ = cv2.findContours(new_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Gambar kontur di atas gambar asli
        cv2.drawContours(new_image, new_contours, -1, (0, 255, 0), 2)

        # Simpan gambar kontur
        cv2.imwrite("static/img/img_now_contours.jpg", new_image)

        # Tentukan ketebalan kontur dengan mengukur luasnya
        new_thickness = []
        for new_contour in new_contours:
            area = cv2.contourArea(new_contour)
            # Misalnya, ambil area yang lebih besar dari 100 piksel sebagai kontur tebal
            if area > 100:
                new_thickness.append(1)
            else:
                new_thickness.append(0)

        # Hasilkan chaincode berdasarkan ketebalan kontur
        new_chain = generate_chaincode(new_thickness)
        return new_chain, original_image
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None
    
def process_image(file_path):
    # Membaca gambar baru
    new_image = cv2.imread(file_path)

    # Mengubah gambar baru menjadi skala abu-abu
    new_gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

    # Terapkan deteksi tepi menggunakan Canny edge detector
    new_edges = cv2.Canny(new_gray, 50, 150)

    # Temukan kontur menggunakan fungsi findContours
    new_contours, _ = cv2.findContours(new_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Tentukan ketebalan kontur dengan mengukur luasnya
    new_thickness = []
    for new_contour in new_contours:
        area = cv2.contourArea(new_contour)
        # Misalnya, ambil area yang lebih besar dari 100 piksel sebagai kontur tebal
        if area > 100:
            new_thickness.append(1)
        else:
            new_thickness.append(0)

    # Hasilkan chaincode berdasarkan ketebalan kontur
    new_chain = generate_chaincode(new_thickness)
    return new_chain

knowledge_base = {
    "disappointed_relieved":[
        [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ],
    "blush": [
         [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ],
   "expressionless":[
       [0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ],
    "face_with_raised_eyebrow":[
        [1],
    ],
    "face_with_rolling_eyes":[
        [0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ],
    "grin":[
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ],
    "grinning":[
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ],
    "heart_eyes":[
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ],
    "hugging_face":[
        [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ],
    "hushed":[
       [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],  
    ],
    "joy":[
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ],
    "kissing":[
        [0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ],
   "kissing_closed_eyes":[
         [0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ],
    "kissing_heart":[
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ],
    " kissing_smiling_eyes":[
        [0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ],
    "laughing":[
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ],
    "neutral_face":[
        [0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ],
    "no_mouth":[
        [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ],
    "open_mouth":[
        [0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ],
    "persevere":[
        [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ],
    "relaxed":[
        [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ],
    "rolling_on_the_floor_laughing":[
        [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ],
    "sleeping":[
        [0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
    ],
    "sleepy":[
        [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ],
    "slightly_smiling_face":[
        [0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ],
    "smile":[
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ],

    "smiley":[
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ],

    "smirk":[
        [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ],
   "star-struck":[
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
    ],
    "sunglasses":[
        [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ],
    "sweat_smile":[
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ],
    "thinking_face":[
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ],
    "tired_face":[
        [0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ],
    "wink":[
        [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ],
    "yum":[
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ],
    "zipper_mouth_face":[
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ],
    # Tambahkan lebih banyak label emoji dengan chaincode mereka
}
def match_chaincode(new_chain, knowledge_base):
    for label, chaincodes in knowledge_base.items():
        for chaincode in chaincodes:
            if new_chain == chaincode:
                return label
    return "Tidak dikenali"
