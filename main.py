import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def main():
    # Load original image
    filename = 'Pictures/original.jpg'
    img = Image.open(filename)
    img.load()

    # Split image into three channels(RGB)
    red, green, blue = img.split()

    # Convert images to numpy arrays
    red = np.array(red)
    green = np.array(green)
    blue = np.array(blue)

    # Param of distortion
    distortion = kernel(15, 5)

    # Param of noise
    threshold = 1e-10

    # Get a blurry image
    noisy_img_r = blur_image(red, distortion)
    noisy_img_g = blur_image(green, distortion)
    noisy_img_b = blur_image(blue, distortion)
    noisy_image = reformation_image(noisy_img_r, noisy_img_g, noisy_img_b, distortion)
    noisy_image.save('Pictures/blur.jpg')

    # Inverse Filtration
    # Use inverse filter for each channel and connect them
    restored_img_r_if = inverse_filtering(noisy_img_r, distortion, threshold=threshold)
    restored_img_g_if = inverse_filtering(noisy_img_g, distortion, threshold=threshold)
    restored_img_b_if = inverse_filtering(noisy_img_b, distortion, threshold=threshold)
    restored_img_if = reformation_image(restored_img_r_if, restored_img_g_if,
                                        restored_img_b_if, distortion)
    restored_img_if.save('Pictures/inverseFiltration.jpg')

    inv_filtration_composition = [noisy_image, restored_img_if, img]
    description_if = ["Blur", "Inverse filtration", "original"]

    draw(inv_filtration_composition, description_if)

    # Wiener Filtration
    # Use wiener filter for each channel and connect them
    wiener_k = 0.00006  # const in wiener expression
    restored_img_r_wiener = wiener_filtration(noisy_img_r, distortion, wiener_k)
    restored_img_g_wiener = wiener_filtration(noisy_img_g, distortion, wiener_k)
    restored_img_b_wiener = wiener_filtration(noisy_img_b, distortion, wiener_k)
    restored_img_wiener = reformation_image(restored_img_r_wiener, restored_img_g_wiener,
                                            restored_img_b_wiener, distortion)

    wiener_filtration_composition = [noisy_image, restored_img_wiener, img]
    description_wiener = ["Blur", "Wiener filtration", "original"]

    draw(wiener_filtration_composition, description_wiener)


# Kernel nucleus
def kernel(size, sigma):
    # Centers of filter
    x0 = size // 2
    y0 = size // 2

    x = np.arange(0, size, dtype=float)
    y = np.arange(0, size, dtype=float)[:, np.newaxis]

    # Exponent part of gaussian function
    exp_part = ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)
    gaussian = 1 / (2 * np.pi * sigma ** 2) * np.exp(-exp_part)

    # Return normalized gaussian func
    return gaussian / np.sum(gaussian)


# Fourier image of distortion func
def get_H(h, shape):
    h_padded = np.zeros(shape)
    h_padded[:h.shape[0], :h.shape[1]] = np.copy(h)
    H = np.fft.fft2(h_padded)
    return H


# Inverse H
def inverse_H(H, threshold=1e-10):
    M = np.zeros(H.shape)
    M[np.abs(H) > threshold] = 1

    H1 = H.copy()
    H1[np.abs(H1) <= threshold] = 1
    return M / H1


def inverse_filtering(blur_img, h, threshold=1e-10):
    G = np.fft.fft2(blur_img)  # FI of blurred img
    H = get_H(h, blur_img.shape)  # H - func of distortion (FI)
    F = G * inverse_H(H, threshold)  # F - func of orig img (FI)
    f = np.fft.ifft2(F)  # orig img
    return np.abs(f)


def blur_image(image, kernel_n):
    F = np.fft.fft2(image)  # F - func of orig img (FI)
    H = get_H(kernel_n, image.shape)  # H - func of distortion (FI)
    G = F * H  # FI of blurred img
    g = np.fft.ifft2(G)  # blurred img
    return np.abs(g)


# r,g,b - numpy arrays
def reformation_image(r, g, b, distortion):
    # Convert numpy array to img
    r = Image.fromarray(r)
    # Convert from F to L format
    r = r.convert('L')

    # The same for each other channels
    g = Image.fromarray(g)
    g = g.convert('L')

    b = Image.fromarray(b)
    b = b.convert('L')

    # Merge channels
    blurry = Image.merge("RGB", (r, g, b))
    # Return blurred img
    return blurry


def wiener_filtration(blur_image, distor_func, K):
    G = np.fft.fft2(blur_image)  # FI of blurred img
    H = get_H(distor_func, blur_image.shape)  # H - func of distortion (FI)
    F = np.conj(H) / (np.abs(H) ** 2 + K) * G  # F - func of orig img (FI)
    restored = np.abs(np.fft.ifft2(F))  # orig img
    return restored


def tile(*images, vertical=False):
    width, height = images[0].width, images[0].height
    tiled_size = (
        (width, height * len(images))
        if vertical
        else (width * len(images), height)
    )
    tiled_img = Image.new(images[0].mode, tiled_size)
    row, col = 0, 0
    for image in images:
        tiled_img.paste(image, (row, col))
        if vertical:
            col += height
        else:
            row += width

    return tiled_img


def draw(image, description):
    fig, ax = plt.subplots(1, len(image))

    for i in range(len(image)):
        ax[i].imshow(image[i])
        ax[i].set_title(description[i])

    fig.set_figwidth(10)
    fig.set_figheight(10)

    plt.show()


if __name__ == '__main__':
    main()
