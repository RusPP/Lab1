import numpy as np
from PIL import Image


def main():
    # Load original image
    filename = "Pictures/original.jpg"
    original_image = Image.open(filename)

    # Split into 3 rgb channels
    red, green, blue = original_image.split()
    red = np.array(red)
    green = np.array(green)
    blue = np.array(blue)

    # Original image in grayscale mode
    original_image_grayscale = original_image.convert('L')
    original_grayscale = np.array(original_image_grayscale)

    # Initialize noise and blur matrix
    noise_intensity = 5
    noise = noise_matrix(original_grayscale.shape, noise_intensity)
    blur = blur_matrix(10, 3, original_grayscale)

    # Blur + noisy
    # Matrices:
    distorted_red = distort_image(red, blur, noise)
    distorted_green = distort_image(green, blur, noise)
    distorted_blue = distort_image(blue, blur, noise)
    # Images:
    distorted_image_red = reform(distorted_red)
    distorted_image_green = reform(distorted_green)
    distorted_image_blue = reform(distorted_blue)
    distorted_image = Image.merge("RGB", (distorted_image_red, distorted_image_green, distorted_image_blue))

    # Inverse filtration
    # Matrices:
    restored_inv_red = inverse_filtration(distorted_red, blur, noise)
    restored_inv_green = inverse_filtration(distorted_green, blur, noise)
    restored_inv_blue = inverse_filtration(distorted_blue, blur, noise)
    # Images:
    restored_image_inv_red = reform(restored_inv_red)
    restored_image_inv_green = reform(restored_inv_green)
    restored_image_inv_blue = reform(restored_inv_blue)
    restored_image_inv = Image.merge("RGB", (restored_image_inv_red, restored_image_inv_green, restored_image_inv_blue))

    # Wiener
    wiener_const = 0.000001
    # Matrices:
    restored_wiener_red = wiener_filtration(distorted_red, blur, wiener_const)
    restored_wiener_green = wiener_filtration(distorted_green, blur, wiener_const)
    restored_wiener_blue = wiener_filtration(distorted_blue, blur, wiener_const)
    # Images:
    restored_image_wiener_red = reform(restored_wiener_red)
    restored_image_wiener_green = reform(restored_wiener_green)
    restored_image_wiener_blue = reform(restored_wiener_blue)
    restored_image_wiener = Image.merge("RGB", (
        restored_image_wiener_red, restored_image_wiener_green, restored_image_wiener_blue))

    # Lucy richardson
    # Matrices:
    iteration = 3
    restored_lucy_red = richardson_lucy(distorted_red, blur, iteration)
    restored_lucy_green = richardson_lucy(distorted_green, blur, iteration)
    restored_lucy_blue = richardson_lucy(distorted_blue, blur, iteration)
    # Images:
    restored_image_lucy_red = reform(restored_lucy_red)
    restored_image_lucy_green = reform(restored_lucy_green)
    restored_image_lucy_blue = reform(restored_lucy_blue)
    restored_image_lucy = Image.merge("RGB",
                                      (restored_image_lucy_red, restored_image_lucy_green, restored_image_lucy_blue))

    # Tikhonov regularization
    # Matrices:
    restored_reg_red = regularization(distorted_red, blur, noise)
    restored_reg_green = regularization(distorted_green, blur, noise)
    restored_reg_blue = regularization(distorted_blue, blur, noise)
    # Images:
    restored_image_reg_red = reform(restored_reg_red)
    restored_image_reg_green = reform(restored_reg_green)
    restored_image_reg_blue = reform(restored_reg_blue)
    restored_image_reg = Image.merge("RGB",
                                     (restored_image_reg_red, restored_image_reg_green, restored_image_reg_blue))

    # Console interactions
    interactive(original_image, distorted_image, restored_image_inv, restored_image_wiener, restored_image_lucy,
                restored_image_reg)


def reform(array):
    image = Image.fromarray(array)
    image = image.convert('L')
    return image


def interactive(original_image, distorted_image, restored_image_inv,
                restored_image_wiener, restored_image_lucy, restored_image_reg):
    description = ["Choose image: ", "0. Original image", "1. Distorted image ", "2. Inverse filtration ",
                   "3. Wiener filtration ",
                   "4. Richardson-Lucy ", "5. Tikhonov regularization ", "6. Close"]
    for i in description:
        print(i)
    work = True
    while work:
        tools = input()
        if tools == '0':
            original_image.show()
        elif tools == '1':
            distorted_image.show()
        elif tools == '2':
            restored_image_inv.show()
        elif tools == '3':
            restored_image_wiener.show()
        elif tools == '4':
            restored_image_lucy.show()
        elif tools == '5':
            restored_image_reg.show()
        elif tools == '6':
            work = False
        else:
            print("Incorrect input")


def distort_image(original, blur, noise):
    blur_spec = np.fft.fft2(blur)
    noise_spec = np.fft.fft2(noise)
    original_spec = np.fft.fft2(original)
    distorted = np.fft.ifft2(original_spec * blur_spec + noise_spec)
    return np.abs(distorted)


def add_blur(image, distortion):
    h = np.fft.fft2(distortion)
    f = np.fft.fft2(image)
    g = h * f
    blurred = np.fft.ifft2(g)
    return np.abs(blurred)


def add_noise(image, factor):
    noise = noise_matrix(image.shape, factor)
    noisy_img = image + noise
    return noisy_img


def blur_matrix(size, sigma, image):
    # Centers of filter
    x0 = size // 2
    y0 = size // 2

    x = np.arange(0, size, dtype=float)
    y = np.arange(0, size, dtype=float)[:, np.newaxis]

    exp_part = ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)
    gaussian = 1 / (2 * np.pi * sigma ** 2) * np.exp(-exp_part)
    gaussian_normalized = gaussian / np.sum(gaussian)

    blur = np.zeros(image.shape)
    blur[:gaussian_normalized.shape[0], :gaussian_normalized.shape[1]] = np.copy(gaussian_normalized)
    return blur


def noise_matrix(shape, var):
    mean = 0.0
    noise = np.random.normal(mean, var, shape)
    noise = noise.reshape(shape)
    return noise


def inverse_filtration(distorted, blur, noise):
    distorted_spec = np.fft.fft2(distorted)
    blur_spec = np.fft.fft2(blur)
    noise_spec = np.fft.fft2(noise)
    restored_spec = distorted_spec / blur_spec + noise_spec / blur_spec
    restored = np.fft.ifft2(restored_spec)
    return np.abs(restored)


def wiener_filtration(distorted, blur, wiener_const):
    distorted_spec = np.fft.fft2(distorted)
    blur_spec = np.fft.fft2(blur)
    restored_spec = np.conj(blur_spec) / (np.abs(blur_spec) ** 2 + wiener_const) * distorted_spec
    restored = np.abs(np.fft.ifft2(restored_spec))
    return restored


def richardson_lucy(distorted, blur, iteration):
    distorted_spec = np.fft.fft2(distorted)
    blur_spec = np.fft.fft2(blur)

    new_restored = distorted
    new_restored_spec = distorted_spec
    for i in range(iteration):
        restored = new_restored
        restored_spec = new_restored_spec
        _hf = np.fft.ifft2(blur_spec * restored_spec)
        _ghf = distorted / _hf
        _hghf = np.fft.ifft2(blur_spec * np.fft.fft2(_ghf))
        new_restored = np.abs(restored * _hghf)
    return new_restored


def regularization(distorted, blur, noise):
    distorted_spec = np.fft.fft2(distorted)
    blur_spec = np.fft.fft2(blur)

    laplas_matrix = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    laplas = np.zeros(distorted.shape)
    laplas[:laplas_matrix.shape[0], :laplas_matrix.shape[1]] = np.copy(laplas_matrix)
    laplas_spec = np.fft.fft2(laplas)

    _lambda = 1.0
    _alpha = 0.1
    step = 0.01
    repeat = 2
    while True:
        restored_spec = np.conj(blur_spec) / (
                np.abs(blur_spec) ** 2 + _lambda * np.abs(laplas_spec) ** 2) * distorted_spec
        restored = np.fft.ifft2(restored_spec)
        discrepancy_spec = distorted_spec - blur_spec * restored_spec
        discrepancy = np.fft.ifft2(discrepancy_spec)
        discrepancy_norm = np.sum(discrepancy ** 2)
        noise_norm = np.sum(noise ** 2)
        if discrepancy_norm < noise_norm - _alpha:
            if repeat == 0:
                step = step/2
            _lambda += step
            repeat = 1
        elif discrepancy_norm > noise_norm + _alpha:
            if repeat == 1:
                step = step/2
            _lambda -= step
            repeat = 0
        else:
            return np.abs(restored)


if __name__ == '__main__':
    main()
