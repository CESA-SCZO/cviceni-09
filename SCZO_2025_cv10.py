# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray

def Exploration_of_neighbourhood(img, seed_map, coordinates, threshold, seed_intensity):
    x = coordinates[0]
    y = coordinates[1]
    new_seeds_temp_coords = []

    if (seed_map[x, y + 1]) == 0:
        if (np.abs(seed_intensity - img[x, y + 1])) <= threshold:
            seed_map[x, y + 1] = 2
            new_seeds_temp_coords.append([x, y + 1])
        else:
            seed_map[x, y + 1] = 1

    if (seed_map[x, y - 1]) == 0:
        if (np.abs(seed_intensity - img[x, y - 1])) <= threshold:
            seed_map[x, y - 1] = 2
            new_seeds_temp_coords.append([x, y - 1])
        else:
            seed_map[x, y - 1] = 1

    if (seed_map[x + 1, y]) == 0:
        if (np.abs(seed_intensity - img[x + 1, y])) <= threshold:
            seed_map[x + 1, y] = 2
            new_seeds_temp_coords.append([x + 1, y])
        else:
            seed_map[x + 1, y] = 1

    if (seed_map[x - 1, y]) == 0:
        if (np.abs(seed_intensity - img[x - 1, y])) <= threshold:
            seed_map[x - 1, y] = 2
            new_seeds_temp_coords.append([x - 1, y])
        else:
            seed_map[x - 1, y] = 1

    return new_seeds_temp_coords, seed_map


def region_growing(img, coordinates, threshold, criterium):
    seed_map = np.zeros_like(img)
    seed_map[:, [0, -1]] = 1
    seed_map[[0, -1], :] = 1
    new_seeds_temp_coords = []
    new_seeds_count = len(coordinates)

    seed_intensity = img[coordinates[0][0], coordinates[0][1]]
    seed_map[coordinates[0][0], coordinates[0][1]] = 2

    while (new_seeds_count != 0):
        new_seeds_coordinates = []
        for i in range(new_seeds_count):
            if criterium == 'dynamic':
                seed_intensity = np.mean(img[seed_map == 2])
            else:
                pass
            new_seeds_temp_coords, seed_map = Exploration_of_neighbourhood(img, seed_map, coordinates[i], threshold,
                                                                           seed_intensity)
            new_seeds_coordinates.extend(new_seeds_temp_coords)
            new_seeds_temp_coords = []
        coordinates = new_seeds_coordinates.copy()
        new_seeds_count = len(coordinates)
    return seed_map

# %% Ukol 1
# =============================================================================
# Načtěte obraz ‘okruh.png’ a převeďte jej do šedotónového
# Nastavte vhodnou pozici semínka na vysegmentování cest okruhu
# Prozkoumejte přiložené funkce region_growing() a Exploration_of_neighbourhood(). Co je vstupem, co výstupem?
# Upravte funkci tak, aby využívala na rozrůstání dynamické kritérium
# Měňte pozice semínka a pozorujte jeho vliv na výsledek segmentace
# =============================================================================

img = rgb2gray(imread('okruh.png')[:, :, :3])

# inicializace segmentace (výstupu region growing):
final_static_segmentation = np.zeros_like(img)
final_dynamic_segmentation = np.zeros_like(img)

# =============================================================================
# STATICKÁ VERZE REGION GROWING
# =============================================================================

# stanovení pozice iniciálního semínka:
seed_x = 369
seed_y = 539
seed_coordinates = [[seed_y, seed_x]]

plt.figure()
plt.imshow(img, cmap='gray')
plt.plot(seed_x, seed_y, 'ro')
plt.show()

# stanovení prahu pro statické kritérium:
static_threshold = 0.18


seed_map_static = region_growing(img, seed_coordinates, static_threshold, criterium='static')

plt.figure()
plt.imshow(seed_map_static, cmap='gray')
plt.show()

final_static_segmentation[seed_map_static == 2] = 1

# pro porovnání - segmentace základním prahováním:
segmentation_mask_thresholding = np.zeros_like(img)
segmentation_mask_thresholding[(img > img[seed_y, seed_x] - static_threshold) & (img < img[seed_y, seed_x] + static_threshold)] = 1

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title('Original')

plt.subplot(1, 3, 2)
plt.imshow(final_static_segmentation, cmap='gray')
plt.axis('off')
plt.title('Segmentace - staticka verze')

plt.subplot(1, 3, 3)
plt.imshow(segmentation_mask_thresholding, cmap='gray')
plt.axis('off')
plt.title('Segmentace - prahovani')
plt.show()

# =============================================================================
# DYNAMICKÁ VERZE REGION GROWING
# =============================================================================

# stanovení pozice iniciálního semínka:
seed_x = 369
seed_y = 539
seed_coordinates = [[seed_y, seed_x]]

# stanovení prahu pro statické kritérium:
dynamic_threshold = 0.18

seed_map_dynamic = region_growing(img, seed_coordinates, dynamic_threshold, 'dynamic')
final_dynamic_segmentation[seed_map_dynamic == 2] = 1

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Původní obraz')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(final_static_segmentation, cmap='gray')
plt.axis('off')
plt.title('Výsledek segmentace - staticky')

plt.subplot(1, 3, 3)
plt.imshow(final_dynamic_segmentation, cmap='gray')
plt.axis('off')
plt.title('Výsledek segmentace - dynamicky')
plt.show()
