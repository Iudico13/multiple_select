import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
import os
from skimage import img_as_ubyte
from skimage.segmentation import clear_border
from skimage.color import label2rgb
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from tkinter import Tk, filedialog
from skimage.io import imread
import shutil
from PIL import Image as pil_Image


# Main loop for execution of multiple-select method
loop = True
while loop:
    print("Multiple-select method\n> Execute steps successively")
    print("1: Define Data set")
    print("2: Testrun")
    print("3: Start complete evaluation")
    print("4: Quit")

    n = int(input("Please choose: "))

    if n == 1:
        print("2: Define Data")
        imageData, Image = define_data()

    elif n == 2:
        print("2: Testrun")
        testRun = True
        angularPos = 1
        imageNumber = 1
        single_run(Image, net, testRun, angularPos, imageNumber, imageData)

    elif n == 3:
        # Segmentation and analysis of all projection images
        print("3: Complete evaluation started")
        testrun = False
        projection_Data_cell = complete_run(imageData, net)

        # Evaluating projection images according to multiple-select approach
        print("3_1: Evaluate projection images")
        mediumDropLengths, accordingProjection = evaluate_droplet_length(
            projection_Data_cell, imageData)

        # Generate directories for each droplet state (length) containing appropriate projection images
        print("3_2: Generate PNG files")
        for state in range(len(mediumDropLengths)):
            generate_projection_folder(accordingProjection, state, mediumDropLengths,
                                                                    imageData)

    elif n == 4:
        loop = False

    else:
        print("Invalid input!")



def define_data():
    print("> Browse to the desired location and select 1 representative projection image")
    print("> Selected projection image must be named in the format scan_0000xxxx_0xxx.png")

    Tk().withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("PNG files", "*.png")])

    file_dir, file_name = os.path.split(file_path)

    # Determination of range of angular positions tested
    name_numberAngPos = file_name.split('_0')[0] + "_0"
    numberAngPos = len([f for f in os.listdir(file_dir) if f.startswith(name_numberAngPos) and f.endswith("_0000.png")])

    # Determination of projection images per angular position
    numberProjIm = len([f for f in os.listdir(file_dir) if f.startswith(name_numberAngPos + "0000000_") and f.endswith(".png")])
    # # ROI Version
    # # Read sample projection image and show
    # samplePI = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    # plt.imshow(samplePI, cmap='gray')
    # plt.title('Sample Projection Image')

    # # Select rectangular ROI
    # print("> Draw rectangle region of interest (ROI) on sample image")
    # print("> Start at top left of ROI and draw to bottom right")
    # print("> ROI must at least contain needle and emerging droplet")

    # rect = plt.ginput(2, timeout=0)
    # imageData = {
    #     'filename': file_name,
    #     'path': file_dir + "/",
    #     'angPos': numberAngPos,
    #     'projPerAng': numberProjIm,
    #     'rect': [rect[0][0], rect[0][1], rect[1][0] - rect[0][0], rect[1][1] - rect[0][1]]
    # }

    # # Check if conditions for ROI are fulfilled
    # while (imageData['rect'][2] < 480 or imageData['rect'][3] < 360 or imageData['rect'][0] < 1 or imageData['rect'][1] < 1):
    #     print("> ROI must be at least 480px x 360px and positions must be positive\n Please try again")
    #     print("> Draw rectangle ROI on sample image\n> Start at top left of ROI and draw to bottom right")
    #     print("> ROI must at least contain needle and emerging droplet")
    #     rect = plt.ginput(2, timeout=0)
    #     imageData['rect'] = [rect[0][0], rect[0][1], rect[1][0] - rect[0][0], rect[1][1] - rect[0][1]]

    # print("ROI ok")
    # plt.close()

    # ImageROI = samplePI[int(imageData['rect'][1]):int(imageData['rect'][1] + imageData['rect'][3]),
    #                       int(imageData['rect'][0]):int(imageData['rect'][0] + imageData['rect'][2])
         
    
    # NO ROI Version
    
    # Read sample projection image and show
    samplePI = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(samplePI, cmap='gray')
    plt.title('Sample Projection Image')
    plt.close()

    # Define Image Data List with Keywords
    # The entire image is considered the ROI
    imageData = {
        'filename': file_name,
        'path': file_dir + "/",
        'angPos': numberAngPos,
        'projPerAng': numberProjIm,
        'rect': [0, 0, samplePI.shape[1], samplePI.shape[0]]  
    }

    ImageROI = samplePI

    return imageData, ImageROI




def single_run(Image, net, testRun, angularPos, imageNumber, imageData):

    #Loading model from hdf5 file
    model = load_model('droplet_model.hdf5')  

    # Preprocess image here as required for model's input
    preprocessed_image = np.expand_dims(normalize(np.array(Image), axis=1),3)

    # Predict using the loaded model
    predictions = model.predict(np.array([preprocessed_image]))


    categoryImg = np.squeeze(predictions)  # remove the singleton dimensions


    # Binarize
    categoryImg_water = (categoryImg == 1)
    categoryImg_needle = (categoryImg == 2)

    # Post-processing
    # Droplets: despeckle, the value of 2000 can be changed to any value
    categoryImg_water_desp = clear_border(categoryImg_water, buffer_size=2000)

    # Needle: keep only the largest segmented area
    label_img = label(categoryImg_needle)
    regions = regionprops(label_img)
    largest_area = max(region.area for region in regions)
    categoryImg_needle = (label_img == largest_area)

    # Morphological operations
    # Filter element, the size of the filter element is 20 and can be changed to any value
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))

    # Water droplets: opening and closing operation
    categoryImg_water_morph = cv2.morphologyEx(categoryImg_water_desp.astype(np.uint8), cv2.MORPH_OPEN, se)
    categoryImg_water_morph = cv2.morphologyEx(categoryImg_water_morph, cv2.MORPH_CLOSE, se)

    # Needle closing operation
    categoryImg_needle_morph = cv2.morphologyEx(categoryImg_needle.astype(np.uint8), cv2.MORPH_CLOSE, se)

    # Find image properties
    projectionData = get_projection_data(categoryImg_water_morph, categoryImg_needle_morph,
                                                              angularPos, imageNumber)

    # Plotting Results
    if testRun:
        plot_test(testImg, categoryImg, categoryImg_water, categoryImg_water_desp, categoryImg_water_morph,
                              projectionData)

    return projectionData



def get_projection_data(Drop_Seg, Needle_Seg, projNumber, imageNumber):
    # Find properties of droplet
    Droplet_centroid = np.array([prop.centroid for prop in regionprops(Drop_Seg)])
    Droplet_BB = np.array([prop.bbox for prop in regionprops(Drop_Seg)])
    Droplet_area = np.array([prop.area for prop in regionprops(Drop_Seg)])

    if len(Droplet_BB) != 0:  # if at least one droplet was found
        # Length of droplet
        Droplet_length = Droplet_BB[:, 3] - Droplet_BB[:, 1]

        # Coordinates of bounding box of droplet
        Droplet_pos_x_top = np.floor(Droplet_BB[:, 0] + (Droplet_BB[:, 2] - Droplet_BB[:, 0]) / 2)
        Droplet_pos_y_top = Droplet_BB[:, 1]
        Droplet_pos_x_bottom = np.floor(Droplet_BB[:, 0] + (Droplet_BB[:, 2] - Droplet_BB[:, 0]) / 2)
        Droplet_pos_y_bottom = np.floor(Droplet_BB[:, 3])

        # Information about angular step and projection number
        projnr = np.ones(len(Droplet_BB)) * projNumber
        imagenr = np.ones(len(Droplet_BB)) * imageNumber

        # Find properties of needle
        Needle_BB = np.array([prop.bbox for prop in regionprops(Needle_Seg)])

        if len(Needle_BB) != 0:  # if needle was found during segmentation using CNN
            Needle_BB_center_pos_x = np.floor(Needle_BB[:, 0] + (Needle_BB[:, 2] - Needle_BB[:, 0]) / 2)
            Needle_BB_center_pos_y = Needle_BB[:, 1]

            if len(Needle_BB_center_pos_y) > 1:  # 2 "cannulas found"
                # Set the lowest area found to be the cannula
                Needle_BB_center_pos_x = Needle_BB_center_pos_x[-1]
                Needle_BB_center_pos_y = Needle_BB_center_pos_y[-1]

            Needle_BB_center_pos_x = np.ones(len(Droplet_BB)) * Needle_BB_center_pos_x
            Needle_BB_center_pos_y = np.ones(len(Droplet_BB)) * Needle_BB_center_pos_y

        else:  # if needle was not found during segmentation using CNN
            Needle_BB_center_pos_x = np.ones(len(Droplet_BB)) * projNumber * 100000000
            Needle_BB_center_pos_y = np.ones(len(Droplet_BB)) * projNumber * 100000000

        projectionData = np.vstack((projnr, imagenr, Droplet_length, Droplet_centroid[:, 1], Droplet_centroid[:, 0], Droplet_pos_x_top, Droplet_pos_y_top, Droplet_pos_x_bottom, Droplet_pos_y_top, Droplet_area, Needle_BB_center_pos_x, Needle_BB_center_pos_y))
    else:  # if no droplet was found
        projectionData = np.zeros(13)

    return projectionData




def complete_run(imageData, net):
    testRun = False
    projectionData = []

    del_ = '_'
    splitString = imageData['filename'].split(del_)

    projection_Data_cell_1 = [None] * imageData['projPerAng']
    projection_Data_cell = [None] * imageData['angPos']
    #for each angular position
    for angularPos in range(imageData['angPos']):
        print(f'Progressing: {angularPos:04d}/{imageData["angPos"]-1:04d}')
        # for each image in the angular position
        for imageNumber in range(imageData['projPerAng']):
            name = f"{splitString[0]}{del_}{angularPos:08d}{del_}{imageNumber:04d}.png"
            Image = imread(os.path.join(imageData['path'], name))
            ImageROI = Image[int(imageData['rect'][1]):int(imageData['rect'][1] + imageData['rect'][3]), int(imageData['rect'][0]):int(imageData['rect'][0] + imageData['rect'][2])]
            #run single_run and temp save projection data
            projectionData_tmp = single_run(ImageROI, net, testRun, angularPos, imageNumber)
            projection_Data_cell_1[imageNumber] = projectionData_tmp

        projection_Data_cell[angularPos] = projection_Data_cell_1.copy()

    np.save(os.path.join(imageData['path'], 'projection_Data_cell.npy'), projection_Data_cell)

    return projection_Data_cell


import numpy as np


def evaluate_droplet_length(projection_Data_cell, imageData):
    searchLength = 55

    distance = []

    for angPos in range(imageData['angPos']):
        distance_row = []
        for pn in range(imageData['projPerAng']):
            array = projection_Data_cell[angPos][pn]
            distances = array[11, :] - array[6, :]
            valid_distances = distances[distances > 0]
            if len(valid_distances) > 0:
                distances1 = np.min(valid_distances)
            else:
                distances1 = np.nan
            distance_row.append(distances1)
        distance.append(distance_row)

    distance = np.array(distance)
    maxVec = np.max(distance[distance < 1200], axis=1)
    minVec = np.min(distance, axis=1)

    numberDroplets = int((np.mean(maxVec) - np.mean(minVec)) / searchLength)
    mediumDropLengths = np.arange(np.mean(minVec) + searchLength / 2, np.mean(maxVec) - searchLength / 2, searchLength)

    accordingProjection = np.zeros((len(mediumDropLengths), imageData['angPos']), dtype=int)

    for j in range(len(mediumDropLengths)):
        for angPos in range(imageData['angPos']):
            delta = np.abs(distance[angPos, :] - mediumDropLengths[j])
            accordingProjection[j, angPos] = np.where(delta == np.min(delta))[0][0]

    return mediumDropLengths, accordingProjection



def generate_projection_folder(according_projection, state, medium_drop_lengths, image_data):
    output_directory = os.path.join(image_data['path'], f'lengthEval_{round(medium_drop_lengths[state]):.4d}')
    os.makedirs(output_directory)

    split_string = image_data['filename'].split('_')

    for j in range(according_projection.shape[1]):
        image_number = according_projection[state, j]
        name = f"{split_string[0]}_{j:08d}_{image_number:04d}.png"
        img_path = os.path.join(image_data['path'], name)
        img = pil_Image.open(img_path)

        name_scan_final = f"{split_string[0]}_{j:08d}.png"
        shutil.copy(os.path.join(image_data['path'], name_scan_final), os.path.join(output_directory, name_scan_final))

        png_file_av = pil_Image.open(os.path.join(output_directory, name_scan_final))
        png_file_av.save(os.path.join(output_directory, name_scan_final), "PNG")

    return



def plot_test(test_img, category_img, category_img_water, category_img_water_desp, category_img_water_morph, data):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    cmap = [
        [86, 180, 233],      # Background
        [213, 94, 0],        # Tube
        [192, 192, 192],     # needle
        [0, 114, 178],       # water
        [240, 228, 66],      # oil
    ]
    cmap = [tuple(color / 255 for color in colors) for colors in cmap]

    image1 = label2rgb(category_img, test_img, colors=cmap, alpha=0.4)
    axes[0, 0].imshow(image1)
    axes[0, 0].set_title('categories found by CNN')

    axes[0, 1].imshow(category_img_water)
    axes[0, 1].set_title('segmented bubbles/slugs')

    axes[1, 0].imshow(category_img_water_desp)
    axes[1, 0].set_title('despeckling')

    axes[1, 1].imshow(category_img_water_morph)
    axes[1, 1].plot(data[3, :], data[4, :], 'g*')
    axes[1, 1].plot(data[5, :], data[6, :], 'c*')
    axes[1, 1].plot(data[7, :], data[8, :], 'b*')
    axes[1, 1].plot(data[10, :], data[11, :], 'r*')
    axes[1, 1].set_title('morphological operations/droplet properties')

    fig.text(0.8, 0.1, 'top needle position', color='red', bbox=dict(edgecolor='gray', facecolor='white'))
    fig.text(0.8, 0.21, 'central droplet positions', color='green', bbox=dict(edgecolor='gray', facecolor='white'))
    fig.text(0.8, 0.32, 'top droplet positions', color='blue', bbox=dict(edgecolor='gray', facecolor='white'))

    plt.show()

