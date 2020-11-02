
'''Oilpainting Effect'''
# import cv2
# img = cv2.imread('reconstructed.jpg')
# res = cv2.xphoto.oilPainting(img, 38, 1)
# res = cv2.resize(res, (960, 540)) 
# cv2.imshow("Frame",res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''Cartoon Effect'''
# import cv2
# img = cv2.imread('gs1.jpg')
# res = cv2.stylization(img, sigma_s=60, sigma_r=0.6)

# cv2.imshow("Frame",res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''Pencil Sketch B/W and coloured '''
# import cv2 
# img = cv2.imread('reconstructed.jpg')
# dst_gray, dst_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.03)
# dst_color = cv2.resize(dst_color, (960, 540)) 
# cv2.imshow("Frame",dst_color)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


'''create_pointillism_art'''
# import scipy.spatial
# import numpy as np
# import random
# import cv2
# import math
# from sklearn.cluster import KMeans

# def compute_color_probabilities(pixels, palette):
#     distances = scipy.spatial.distance.cdist(pixels, palette)
#     maxima = np.amax(distances, axis=1)
#     distances = maxima[:, None] - distances
#     summ = np.sum(distances, 1)
#     distances /= summ[:, None]
#     return distances

# def get_color_from_prob(probabilities, palette):
#     probs = np.argsort(probabilities)
#     i = probs[-1]
#     return palette[i]

# def randomized_grid(h, w, scale):
#     assert (scale > 0)
#     r = scale//2
#     grid = []
#     for i in range(0, h, scale):
#         for j in range(0, w, scale):
#             y = random.randint(-r, r) + i
#             x = random.randint(-r, r) + j
#     grid.append((y % h, x % w))
#     random.shuffle(grid)
#     return grid

# def get_color_palette(img, n=25):
#     clt = KMeans(n_clusters=n)
#     clt.fit(img.reshape(-1, 3))
#     return clt.cluster_centers_

# def complement(colors):
#     return 255 - colors

# def create_pointillism_art(image_path, primary_colors):
        
#     img = cv2.imread(image_path)
#     radius_width = int(math.ceil(max(img.shape) / 1500))
#     palette = get_color_palette(img, primary_colors)
#     complements = complement(palette)
#     palette = np.vstack((palette, complements))
#     canvas = img.copy()
#     grid = randomized_grid(img.shape[0], img.shape[1], scale=7)
    
#     pixel_colors = np.array([img[x[0], x[1]] for x in grid])
    
#     color_probabilities = compute_color_probabilities(pixel_colors, palette)


#     for i, (y, x) in enumerate(grid):
#         color = get_color_from_prob(color_probabilities[i], palette)
#         cv2.ellipse(canvas, (x, y), (radius_width, radius_width), 0, 0, 360, color, -1, cv2.LINE_AA)
    
#     return canvas

# res = create_pointillism_art('gs1.jpg', 10)
# cv2.imshow("Frame",res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''Details Enhancement'''
# import cv2
# src = cv2.imread('gs1.jpg')
# res = cv2.detailEnhance(src, sigma_s=10, sigma_r=0.15)
# cv2.imshow("Frame",res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''edge Preserving Filter '''
# import cv2
# src = cv2.imread('gs1.jpg')
# res = cv2.edgePreservingFilter(src, flags=1, sigma_s=60, sigma_r=0.4)
# cv2.imshow("Frame",res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import numpy as np
# import argparse
# import cv2
# import requests
# import random

# def closest_color(i):
#     r, g, b = 0, 0, 0
#     d = 20000
#     r1,g1,b1 = i[2],i[1],i[0]
#     for j in pallete:
#         r2,g2,b2 = j[2],j[1],j[0]
#         if d > ((r2-r1)*0.30) ** 2 + ((g2-g1)*0.59) ** 2 + ((b2-b1)*0.11) ** 2:
#             if random.randint(0,255)%2:
#                 r, g, b = r2, g2, b2
#             d = ((r2-r1)*0.30) ** 2 + ((g2-g1)*0.59) ** 2 + ((b2-b1)*0.11) ** 2
#     return (r,g,b)

# # ap = argparse.ArgumentParser()
# # ap.add_argument("-i", "--image", help = "path to the image")
# # args = vars(ap.parse_args())
 
# image = cv2.imread('gs1.jpg',cv2.IMREAD_COLOR)
# r = requests.get("http://api.noopschallenge.com/hexbot?count=40")

# h,w,x = image.shape

# pallete = []
# for i in r.json()['colors']:
#     pallete.append((int(i['value'][1:3], 16), int(
#         i['value'][3:5], 16), int(i['value'][5:7], 16)))
# # print (pallete)

# canv = np.full((h, w, 3), 0, dtype=np.uint8)
# # print (pallete)
# for i in range(h):
#     for j in range(w):
#         try:
#             t = closest_color(image[j][i])
#             cv2.circle(canv,(i,j),3,t)
#         except:
#             continue
#     # print (h - i)
# cv2.imshow("image", image)
# cv2.imshow("image1", canv)
# cv2.waitKey(0)