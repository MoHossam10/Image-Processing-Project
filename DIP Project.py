#imports
import pygame
from pygame.locals import *
import cv2
import numpy as np
from matplotlib import pyplot as pit
import math
import matplotlib.pyplot as plt


# Initialize Pygame
pygame.init()

# Set up the screen
WIDTH, HEIGHT = 1500, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("FILTERS APP")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
RED = (255, 0, 0)

# Fonts
font = pygame.font.Font(None, 28)



def main():
    clock = pygame.time.Clock()

    # Initialize variables
    flag=0
    # Load the image using OpenCV
    loaded_image = cv2.imread(r"C:\Users\pc\OneDrive\Desktop\hossamghaly.jpg",0)
    cv2.imshow("Orignal Image",loaded_image)
    copy_image = loaded_image.copy()  # Create a copy for filtering



    # Smoothing filters
    median_rect = pygame.Rect(30, 200, 90, 40)
    adaptive_median_rect = pygame.Rect(130, 200, 180, 40)
    averaging_rect = pygame.Rect(320, 200, 120, 40)
    gaussian_rect = pygame.Rect(450, 200, 120, 40)

    # Sharpening filters
    Laplacian_rect = pygame.Rect(30, 290, 120, 40)
    Unsharp_rect = pygame.Rect(160, 290, 100, 40)
    Roberts_rect = pygame.Rect(270, 290, 100, 40)
    Sobel_rect = pygame.Rect(380, 290, 80, 40)

    # Noise filters
    Impulse_rect = pygame.Rect(30, 380, 100, 40)
    Gaussian_rect = pygame.Rect(140, 380, 110, 40)
    Uniform_rect = pygame.Rect(260, 380, 100, 40)

    # Transform /Frequency Domain filters
    histo_equ_rect = pygame.Rect(30, 470, 110, 40)
    Fourier_rect = pygame.Rect(150, 470, 90, 40)
    Interpolation_NN_rect = pygame.Rect(250, 470, 180, 40)
    Interpolation_bilinear_rect = pygame.Rect(440, 470, 220, 40)







    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_pos = pygame.mouse.get_pos()
                    if median_rect.collidepoint(mouse_pos):
                        padding_value = 10
                        padding_image = []
                        height, width = copy_image.shape
                        for i in range(height + 2 * padding_value):
                            row_pixels = []
                            for j in range(width + 2 * padding_value):
                                if i >= padding_value and i < height + padding_value and j >= padding_value and j < width + padding_value:
                                    row_pixels.append(copy_image[i - padding_value, j - padding_value])
                                else:
                                    row_pixels.append(0)
                            padding_image.append(row_pixels)

                        padding_image = np.array(padding_image, dtype=np.uint8)
                        filtered_image = apply_median_filter(padding_image)
                        flag=1
                        copy_image = loaded_image.copy()
                    if Gaussian_rect.collidepoint(mouse_pos):
                        padding_value = 10
                        padding_image = []
                        height, width = copy_image.shape
                        for i in range(height + 2 * padding_value):
                            row_pixels = []
                            for j in range(width + 2 * padding_value):
                                if i >= padding_value and i < height + padding_value and j >= padding_value and j < width + padding_value:
                                    row_pixels.append(copy_image[i - padding_value, j - padding_value])
                                else:
                                    row_pixels.append(0)
                            padding_image.append(row_pixels)

                        padding_image = np.array(padding_image, dtype=np.uint8)
                        filtered_image = guasssian_noise(padding_image)
                        flag=2
                        copy_image = loaded_image.copy()
                    if Roberts_rect.collidepoint(mouse_pos):
                        padding_value = 10
                        padding_image = []
                        height, width = copy_image.shape
                        for i in range(height + 2 * padding_value):
                            row_pixels = []
                            for j in range(width + 2 * padding_value):
                                if i >= padding_value and i < height + padding_value and j >= padding_value and j < width + padding_value:
                                    row_pixels.append(copy_image[i - padding_value, j - padding_value])
                                else:
                                    row_pixels.append(0)
                            padding_image.append(row_pixels)

                        padding_image = np.array(padding_image, dtype=np.uint8)
                        filtered_image = roberts_cross_gradient(padding_image)
                        flag=3
                        copy_image = loaded_image.copy()
                    if gaussian_rect.collidepoint(mouse_pos):
                        padding_value = 10
                        padding_image = []
                        height, width = copy_image.shape
                        for i in range(height + 2 * padding_value):
                            row_pixels = []
                            for j in range(width + 2 * padding_value):
                                if i >= padding_value and i < height + padding_value and j >= padding_value and j < width + padding_value:
                                    row_pixels.append(copy_image[i - padding_value, j - padding_value])
                                else:
                                    row_pixels.append(0)
                            padding_image.append(row_pixels)

                        padding_image = np.array(padding_image, dtype=np.uint8)
                        filtered_image = apply_gaussian_filter(padding_image)
                        flag=4
                        copy_image = loaded_image.copy()
                    if averaging_rect.collidepoint(mouse_pos):
                        padding_value = 10
                        padding_image = []
                        height, width = copy_image.shape
                        for i in range(height + 2 * padding_value):
                            row_pixels = []
                            for j in range(width + 2 * padding_value):
                                if i >= padding_value and i < height + padding_value and j >= padding_value and j < width + padding_value:
                                    row_pixels.append(copy_image[i - padding_value, j - padding_value])
                                else:
                                    row_pixels.append(0)
                            padding_image.append(row_pixels)

                        padding_image = np.array(padding_image, dtype=np.uint8)
                        filtered_image = averaging_filter(padding_image)
                        flag=5
                        copy_image = loaded_image.copy()
                    if Unsharp_rect.collidepoint(mouse_pos):
                        padding_value = 10
                        padding_image = []
                        height, width = copy_image.shape
                        for i in range(height + 2 * padding_value):
                            row_pixels = []
                            for j in range(width + 2 * padding_value):
                                if i >= padding_value and i < height + padding_value and j >= padding_value and j < width + padding_value:
                                    row_pixels.append(copy_image[i - padding_value, j - padding_value])
                                else:
                                    row_pixels.append(0)
                            padding_image.append(row_pixels)

                        padding_image = np.array(padding_image, dtype=np.uint8)
                        filtered_image = unsharp_filter(padding_image)
                        flag=6
                        copy_image = loaded_image.copy()
                    if Impulse_rect.collidepoint(mouse_pos):
                        filtered_image = impluse_filter(copy_image)
                        flag=7
                        copy_image = loaded_image.copy()
                    if Interpolation_NN_rect.collidepoint(mouse_pos):
                        filtered_image = NN_filter(copy_image)
                        flag=8
                        copy_image = loaded_image.copy()
                    if Interpolation_bilinear_rect.collidepoint(mouse_pos):
                        filtered_image = bilinear_filter(copy_image)
                        flag=9
                        copy_image = loaded_image.copy()
                    if Laplacian_rect.collidepoint(mouse_pos):
                        filtered_image = Laplacian_Operator(copy_image)
                        flag=10
                        copy_image = loaded_image.copy()
                    if Sobel_rect.collidepoint(mouse_pos):
                        filtered_image = sobel_operator(copy_image)
                        flag=11
                        copy_image = loaded_image.copy()
                    if Uniform_rect.collidepoint(mouse_pos):
                        filtered_image = uniform_noise(copy_image)
                        flag=12
                        copy_image = loaded_image.copy()
                    if adaptive_median_rect.collidepoint(mouse_pos):
                        filtered_image = Adaptive_filter(copy_image)
                        flag=13
                        copy_image = loaded_image.copy()
                    if Fourier_rect.collidepoint(mouse_pos):
                        filtered_image = Fourier_Transform(copy_image)
                        flag=14
                        copy_image = loaded_image.copy()
                    if histo_equ_rect.collidepoint(mouse_pos):
                        filtered_image = equalizeHistogram(copy_image)
                        filtered_image = float2int(filtered_image)
                        flag=15
                        copy_image = loaded_image.copy()




        screen.fill(WHITE)
        # draw rectangles to separate the screen
        pygame.draw.rect(screen, BLACK, pygame.Rect(20, 20, 780, 600), 3)
        draw_text(screen, "Click on Filter:", (30, 30))
        draw_text(screen, "Orignal Image:", (810, 0))
        draw_text(screen, "Filterd Image:", (810, 270))
        draw_text(screen, "Smoothing Filters:", (30, 170))
        draw_text(screen, "Sharpening Filters:", (30, 260))
        draw_text(screen, "Noise Filters:", (30, 350))
        draw_text(screen, "Transform /Frequency Domain Filters:", (30, 440))



        # Draw the filters button

        #Smoothing filters
        pygame.draw.rect(screen, GRAY, median_rect)
        draw_text(screen, "Median", (median_rect.x + 10, median_rect.y + 10))

        pygame.draw.rect(screen, GRAY, adaptive_median_rect)
        draw_text(screen, "Adaptive median", (adaptive_median_rect.x + 10, adaptive_median_rect.y + 10))

        pygame.draw.rect(screen, GRAY, averaging_rect)
        draw_text(screen, "Averaging", (averaging_rect.x + 10, averaging_rect.y + 10))

        pygame.draw.rect(screen, GRAY, gaussian_rect)
        draw_text(screen, "Gaussian", (gaussian_rect.x + 10, gaussian_rect.y + 10))


        #Sharpening filters
        pygame.draw.rect(screen, GRAY, Laplacian_rect)
        draw_text(screen, "Laplacian", (Laplacian_rect.x + 10, Laplacian_rect.y + 10))

        pygame.draw.rect(screen, GRAY, Unsharp_rect)
        draw_text(screen, "Unsharp", (Unsharp_rect.x + 10, Unsharp_rect.y + 10))

        pygame.draw.rect(screen, GRAY, Roberts_rect)
        draw_text(screen, "Roberts", (Roberts_rect.x + 10, Roberts_rect.y + 10))

        pygame.draw.rect(screen, GRAY, Sobel_rect)
        draw_text(screen, "Sobel", (Sobel_rect.x + 10, Sobel_rect.y + 10))


        # Noise filters
        pygame.draw.rect(screen, GRAY, Impulse_rect)
        draw_text(screen, "Impulse", (Impulse_rect.x + 10, Impulse_rect.y + 10))

        pygame.draw.rect(screen, GRAY, Gaussian_rect)
        draw_text(screen, "Gaussian", (Gaussian_rect.x + 10, Gaussian_rect.y + 10))

        pygame.draw.rect(screen, GRAY, Uniform_rect)
        draw_text(screen, "Uniform", (Uniform_rect.x + 10, Uniform_rect.y + 10))


        # Transform /Frequency Domain filters
        pygame.draw.rect(screen, GRAY, histo_equ_rect)
        draw_text(screen, "Histo Equ", (histo_equ_rect.x + 10, histo_equ_rect.y + 10))



        pygame.draw.rect(screen, GRAY, Fourier_rect)
        draw_text(screen, "Fourier", (Fourier_rect.x + 10, Fourier_rect.y + 10))

        pygame.draw.rect(screen, GRAY, Interpolation_NN_rect)
        draw_text(screen, "Interpolation(NN)", (Interpolation_NN_rect.x + 10, Interpolation_NN_rect.y + 10))

        pygame.draw.rect(screen, GRAY, Interpolation_bilinear_rect)
        draw_text(screen, "Interpolation(bilinear)", (Interpolation_bilinear_rect.x + 10, Interpolation_bilinear_rect.y + 10))




        # Display the filtered image
        if flag==1:
            cv2.imshow("Median Image", filtered_image)

        if flag==2:
            cv2.imshow("Gaussian Noise Image", filtered_image)

        if flag==3:
            cv2.imshow("Roberts Image", filtered_image)

        if flag==4:
            cv2.imshow("Gaussian Filter Image", filtered_image)

        if flag==5:
            cv2.imshow("Averaging Image", filtered_image)

        if flag==6:
            cv2.imshow("Unsharp Image", filtered_image)

        if flag==7:
            cv2.imshow("Impluse Image", filtered_image)

        if flag==8:
            cv2.imshow("NN Image", filtered_image)

        if flag==9:
            cv2.imshow("Bilinear Image", filtered_image)

        if flag==10:
            cv2.imshow("Laplacian Image", filtered_image)

        if flag==11:
            cv2.imshow("Sobel Image", filtered_image)

        if flag==12:
            cv2.imshow("Uniform Image", filtered_image)

        if flag==13:
            cv2.imshow("Adaptive median Image", filtered_image)

        if flag==14:
            cv2.imshow("Fourier Image", filtered_image)

        if flag==15:
            cv2.imshow("Histo Equ Image", filtered_image)




        pygame.display.flip()
        clock.tick(30)


def draw_text(surface, text, pos):
    lines = text.split('\n')
    y = pos[1]
    for line in lines:
        text_surface = font.render(line, True, BLACK)
        surface.blit(text_surface, (pos[0], y))
        y += font.get_height() + 5  # Add some spacing between lines




def apply_median_filter(image):
    filtered_image = np.zeros(image.shape, dtype=np.uint8)
    height, width = image.shape
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Extract the 3x3 neighborhood
            neighborhood = image[i - 1:i + 2, j - 1:j + 2]
            # Flatten the neighborhood to a 1D array and sort it
            sorted_pixels = np.sort(neighborhood.flatten())
            # Select the median value
            median_value = sorted_pixels[len(sorted_pixels) // 2]
            # Assign the median value to the corresponding pixel
            filtered_image[i, j] = median_value
    return filtered_image




def guasssian_noise(image):
    mean = 0
    std_dev = 30  # You can adjust this value to control the amount of noise
    noise = np.random.normal(mean, std_dev, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image




def roberts_cross_gradient(image):
    # Create Roberts kernels
    Gx = np.array([[1, 0],
                   [0, -1]])
    Gy = np.array([[0, 1],
                   [-1, 0]])
    # Create output image
    output_image = np.zeros_like(image)
    # Iterate over each pixel, excluding border pixels
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            # Compute gradient in x-direction (Gx)
            gx = np.sum(np.multiply(Gx, image[i-1:i+1, j-1:j+1]))
            # Compute gradient in y-direction (Gy)
            gy = np.sum(np.multiply(Gy, image[i-1:i+1, j-1:j+1]))
            # Compute gradient magnitude
            gradient_magnitude = np.sqrt(gx**2 + gy**2)
            # Assign gradient magnitude to output image
            output_image[i, j] = gradient_magnitude
    return output_image




def apply_gaussian_filter(image):
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]])

    # Create output image
    filtered_image = np.empty_like(image)
    filtered_image[:] = image
    height, width = filtered_image.shape
    padding_value = 1
    # Perform convolution
    for r in range(padding_value, height - padding_value):
        for c in range(padding_value, width - padding_value):
            total = 0

            for i in range(-1, 2):
                for j in range(-1, 2):
                    total += image[r + i, c + j] * kernel[i + 1, j + 1]

            filtered_image[r, c] = total / np.sum(kernel)

    return filtered_image



def averaging_filter(image):
    height, width = image.shape
    local_summition = 0
    padding_value = 10
    mean_image = np.zeros((height, width), np.uint8)
    for i in range(padding_value, height - padding_value):
        for j in range(padding_value, width - padding_value):
            local_summition += image[i - 1][j - 1]
            local_summition += image[i - 1][j]
            local_summition += image[i - 1][j + 1]
            local_summition += image[i][j - 1]
            local_summition += image[i][j]
            local_summition += image[i][j + 1]
            local_summition += image[i + 1][j - 1]
            local_summition += image[i + 1][j]
            local_summition += image[i + 1][j + 1]
            local_mean = local_summition / 9
            mean_image[i - padding_value, j - padding_value] = local_mean
            # print(local_mean)
            local_summition = 0
            local_mean = 0
    return mean_image



def unsharp_filter(image):
    height, width = image.shape
    padding_value = 1
    kernel = np.ones((3, 3), np.float32) / 9
    blurred_image = np.zeros((height, width), np.uint8)
    for i in range(padding_value, height - padding_value):
        for j in range(padding_value, width - padding_value):
            output_pixel = np.sum(image[i - 1:i + 2, j - 1:j + 2] * kernel)
            blurred_image[i - padding_value, j - padding_value] = np.clip(output_pixel, 0, 255)

    mask_image = cv2.subtract(image, blurred_image)
    sharpen_image = cv2.add(mask_image, image)
    return sharpen_image



def impluse_filter(image):
    height, width = image.shape
    noised_image = np.zeros((height, width), dtype=np.float32)

    pepper = 0.25
    salt = 0.9

    for i in range(height):
        for j in range(width):
            rdn = np.random.random()
            if rdn < pepper:
                noised_image[i][j] = 0
            elif rdn > salt:
                noised_image[i][j] = 1
            else:
                noised_image[i][j] = image[i][j]

    return noised_image



def NN_filter(image):
    height, width = image.shape
    interpolated_image = np.zeros((height * 2, width * 2), np.uint8)

    for i in range(height):
        for j in range(width):

            intensity = image[i, j]

            # Assign the intensity value to the corresponding locations in the new image
            interpolated_image[i * 2, j * 2] = intensity
            interpolated_image[i * 2 + 1, j * 2] = intensity
            interpolated_image[i * 2, j * 2 + 1] = intensity
            interpolated_image[i * 2 + 1, j * 2 + 1] = intensity

    return interpolated_image


def bilinear_filter(image):
    height, width = image.shape
    scale_x = width / (width * 2)
    scale_y = height / (height * 2)
    interpolated_image = np.zeros((height * 2, width * 2), np.uint8)


    for i in range(height * 2):
        for j in range(width * 2):

            x = j * scale_x
            y = i * scale_y


            x1 = int(x)
            x2 = min(x1 + 1, width - 1)
            y1 = int(y)
            y2 = min(y1 + 1, height - 1)


            dx = x - x1
            dy = y - y1

            interpolated_value = (1 - dx) * (1 - dy) * image[y1, x1] + dx * (1 - dy) * image[y1, x2] + (1 - dx) * dy * \
                                 image[y2, x1] + dx * dy * image[y2, x2]

            interpolated_image[i, j] = interpolated_value

    return interpolated_image




def Laplacian_Operator(image):
    height, width = image.shape
    kernel = np.array([[0, -1, 0],
                        [-1, 4, -1],
                        [0, -1, 0]])
    filtered_image = np.zeros_like(image, dtype=np.float32)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            window = image[i-1:i+2, j-1:j+2]
            filtered_pixel = np.sum(window * kernel)
            filtered_image[i, j] = filtered_pixel
# Convert the result to uint8
    filtered_image = np.uint8(np.absolute(filtered_image))
    return filtered_image



def sobel_operator(image):
    height, width = image.shape
    x_kernel = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
    y_kernel = np.array([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]])
    sobelx = np.zeros_like(image, dtype=np.float32)
    sobely = np.zeros_like(image, dtype=np.float32)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            window = image[i-1:i+2, j-1:j+2]
            sobelx[i, j] = np.sum(window * x_kernel)
            sobely[i, j] = np.sum(window * y_kernel)
    # Convert the results to absolute values
    sobelx = np.abs(sobelx)
    sobely = np.abs(sobely)
    # Combine the horizontal and vertical gradients
    sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
    # Scale the result to 0-255
    sobel = ((sobel - np.min(sobel)) / (np.max(sobel) - np.min(sobel))) * 255
    sobel = sobel.astype(np.uint8)
    return sobel



def uniform_noise(image):
  noise = np.random.uniform(low=0, high=50, size=image.shape).astype(np.uint8)
  noisy_image = cv2.add(image, noise)

  # Define the filter size (e.g., 3x3 window)
  filter_size = 3

  # Create an empty output image
  filtered_image = np.zeros_like(image)

  # Apply the uniform noise filter
  for i in range(filter_size//2, image.shape[0] - filter_size//2):
      for j in range(filter_size//2, image.shape[1] - filter_size//2):
          window = noisy_image[i-filter_size//2:i+filter_size//2+1, j-filter_size//2:j+filter_size//2+1]
          filtered_image[i, j] = np.mean(window)

  # Convert the result to uint8
  filtered_image = filtered_image.astype(np.uint8)
  return filtered_image


def Adaptive_filter(image):
    height, width = image.shape
    filterd_image = np.zeros_like(image, dtype=np.float32)
    padded_image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            window = padded_image[i - 1:i + 2, j - 1:j + 2]
            d = 2
            filtered_pixel = window.copy().flatten()
            while len(window) <= 7:
                window = padded_image[i - d - 1:i + d, j - d - 1:j + d]
                Zmin = np.min(filtered_pixel)
                Zmax = np.max(filtered_pixel)
                zmed= filtered_pixel//2
                A1 = Zmed - Zmin
                A2 = Zmed - Zmax
                if A1 > 0 and A2 < 0:
                    B1 = image[i][j] - Zmin
                    B2 = image[i][j] - Zmax
                    if B1 > 0 and B2 < 0:
                        filterd_image[i][j] = padded_image[i][j]

                        break
                    else:
                        filterd_image[i][j] = Zmed

                        break
                else:
                    d += 1
    return padded_image


def Fourier_Transform(image):
  fft = np.fft.fft2(image)
  fft_shift = np.fft.fftshift(fft)
  rows, cols = image.shape
  center_row, center_col = rows // 2, cols // 2
  radius = 50
  mask = np.zeros((rows, cols), dtype=np.uint8)
  mask = cv2.circle(mask, (center_col, center_row), radius, 1, -1)
  filtered_shift = fft_shift * mask
  filtered = np.fft.ifftshift(filtered_shift)
  filtered_image = np.abs(np.fft.ifft2(filtered))
  return filtered_image





def float2int(img):
    img = np.round(img, 0)
    img = np.minimum(img, 255)
    img = np.maximum(img, 0)
    img = img.astype('uint8')

    return img
def equalizeHistogram(img):
    img_height = img.shape[0]
    img_width = img.shape[1]
    histogram = np.zeros([256], np.int32)

    # calculate histogram
    for i in range(0, img_height):
        for j in range(0, img_width):
            histogram[img[i, j]] += 1

    # calculate pdf of the image
    pdf_img = histogram / histogram.sum()

    # calculate CDF
    cdf = np.zeros(256, float)
    cdf[0] = pdf_img[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + pdf_img[i]

    cdf_eq = np.round(cdf * 255, 0)  # mapping, transformation function T(x)

    imgEqualized = np.zeros((img_height, img_width))

    # for mapping input image to s.
    for i in range(0, img_height):
        for j in range(0, img_width):
            r = img[i, j]  # feeding intensity levels of pixels into r.
            s = cdf_eq[r]  # finding value of s by finding r'th position in the cdf_eq list.
            imgEqualized[i, j] = s  # mapping s thus creating new output image.


    return imgEqualized


if __name__ == '__main__':
    main()

# Close the OpenCV windows
cv2.waitKey(0)
cv2.destroyAllWindows()
