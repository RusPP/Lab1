import numpy as np
from PIL import Image


def main():
    filename = "Pictures/original.jpg"
    original_image = Image.open(filename)
    original_image = original_image.convert('L')
    original = np.array(original_image)

    noise = noise_matrix(original.shape, 0.01)
    blur = blur_matrix(10, 3, original)

    distorted = distort_image(original, blur, noise)
    distorted_image = Image.fromarray(distorted)

    restored_inv = inverse_filtration(distorted, blur, noise)
    restored_image_inv = Image.fromarray(restored_inv)

    wiener_const = 0.000001
    restored_wiener = wiener_filtration(distorted, blur, wiener_const)
    restored_image_wiener = Image.fromarray(restored_wiener)

    interactive(distorted_image, restored_image_inv, restored_image_wiener)


def interactive(distorted_image, restored_image_inv, restored_image_wiener):
    description = ["Choose image: ", "1. Distorted image ", "2. Inverse filtration ", "3. Wiener filtration ",
                   "0. Close"]
    for i in description:
        print(i)
    work = True
    while work:
        tools = input()
        if tools == '1':
            distorted_image.show()
        elif tools == '2':
            restored_image_inv.show()
        elif tools == '3':
            restored_image_wiener.show()
        elif tools == '0':
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


if __name__ == '__main__':
    main()
