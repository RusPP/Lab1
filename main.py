import numpy as np
from PIL import Image


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

    # Params of distortion
    distortion = kernel(15, 5)

    # Get a blurry image
    noisy_img_r = blur_image(red, distortion)
    noisy_img_g = blur_image(green, distortion)
    noisy_img_b = blur_image(blue, distortion)
    noisy_image = reformation_image(noisy_img_r, noisy_img_g, noisy_img_b, distortion)
    noisy_image.save('Pictures/blur.jpg')

    # Inverse Filtration
    restored_img_r = inverse_filtering(noisy_img_r, distortion)
    restored_img_g = inverse_filtering(noisy_img_g, distortion)
    restored_img_b = inverse_filtering(noisy_img_b, distortion)
    restored_img = reformation_image(restored_img_r, restored_img_g, restored_img_b, distortion)
    restored_img.save('Pictures/inverseFiltration.jpg')

    tile(noisy_image, restored_img, img).show()

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


if __name__ == '__main__':
    main()
