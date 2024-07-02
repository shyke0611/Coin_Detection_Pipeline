# Built in packages
from collections import deque
import math
import sys

# Matplotlib will need to be installed if it isn't already. This is the only package allowed for this base part of the 
# assignment.
from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png

# Define constant and global variables
TEST_MODE = False    # Please, DO NOT change this variable!

def readRGBImageToSeparatePixelArrays(input_filename):
    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)
        

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)

# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):
    new_pixel_array = []
    for _ in range(image_height):
        new_row = []
        for _ in range(image_width):
            new_row.append(initValue)
        new_pixel_array.append(new_row)

    return new_pixel_array


# STEP 1)

# FIRST PART convert RGB image to greyscale

def convertRGBToGreyscale(image_height,image_width,px_array_r, px_array_g, px_array_b):
    # Initialize grayscale pixel array
    px_array_grey = createInitializedGreyscalePixelArray(image_width, image_height)
    # loop through each pixel in image
    for i in range(image_height):
        for j in range(image_width):
            # Calculate grayscale value using specified ratios
            grey_value = round(0.3 * px_array_r[i][j] + 0.6 * px_array_g[i][j] + 0.1 * px_array_b[i][j])
            px_array_grey[i][j] = grey_value
    
    return px_array_grey


#SECOND PART Constrast stretching

# calculate the histogram
def calculateHistogram(px_array):
    # initialise histogram list
    histogram = [0] * 256
    # fill in pixel values into the 1d list
    for row in px_array:
        for pixel in row:
            histogram[int(pixel)] += 1
    return histogram

# calculate the cumulative histogram
def calculateCumulativeHistogram(histogram):
    # initialise cumulative histogram
    cumulative_histogram = [0] * 256
    # Calculate cumulative histogram
    cumulative_sum = 0
    for i in range(len(cumulative_histogram)):
        cumulative_sum += histogram[i]
        cumulative_histogram[i] = cumulative_sum

    return cumulative_histogram


# calculate the 5 and 95% percentiles(fmin,fmax)
def findPercentiles(cumulative_histogram, total_pixels):
    five_percentile_index = int(0.05 * total_pixels)
    ninety_five_percentile_index = int(0.95 * total_pixels)
    
    # find fmin
    i = 0
    while cumulative_histogram[i] < five_percentile_index:
        i += 1
    f_min = i

    # find fmax
    i = 255
    while cumulative_histogram[i] > ninety_five_percentile_index:
        i -= 1
    f_max = i
    
    return f_min, f_max

# performing linear mapping
def stretchPixels(px_array, f_min, f_max):
    stretched_px_array = []
    
    for row in px_array:
        scaled_row = []
        # Perform linear mapping calculation for each pixel in the row
        for pixel in row:
            scaled_pixel = (255 / (f_max - f_min)) * (pixel - f_min)
            scaled_row.append(scaled_pixel)
        
        stretched_row = []
        # Loop through the scaled row and make values within to the range [0, 255]
        for pixel in scaled_row:
            rounded_pixel = round(pixel)
            range_pixel = min(255, max(0, rounded_pixel))
            stretched_row.append(range_pixel)
        
        stretched_px_array.append(stretched_row)
    
    return stretched_px_array

# putting all the function together
def stretchGreyScaleValues(px_array):
    histogram = calculateHistogram(px_array)
    cumulative_histogram = calculateCumulativeHistogram(histogram)
    total_pixels = sum(histogram)
    f_min, f_max = findPercentiles(cumulative_histogram, total_pixels)
    print("Minimum pixel value after stretching:", f_min)
    print("Maximum pixel value after stretching:", f_max)
    stretched_px_array = stretchPixels(px_array, f_min, f_max)
    return stretched_px_array







# STEP 2) Edge Detection


# Part 1) Apply a 3x3 Scharr filter  in x and y

def applyScharrFilter(px_array):
    scharr_x = [
        [3/32, 0, -3/32],
        [10/32, 0, -10/32],
        [3/32, 0, -3/32]
    ]

    scharr_y = [
        [3/32, 10/32, 3/32],
        [0, 0, 0],
        [-3/32, -10/32, -3/32]
    ]
    
    inverted_scharr_x = [row[::-1] for row in scharr_x[::-1]]


    width, height = len(px_array[0]), len(px_array)
    edges_x = createInitializedGreyscalePixelArray(width, height)
    edges_y = createInitializedGreyscalePixelArray(width, height)
    
    for i in range(1, height - 1):
     for j in range(1, width - 1):
        gx = 0
        gy = 0
        
        for m in range(3):
            for n in range(3):
                # Calculate the index offsets
                x_offset = i + m - 1
                y_offset = j + n - 1
                
                # Apply the inverted Scharr filter for gx
                gx += inverted_scharr_x[m][n] * px_array[x_offset][y_offset]
                
                # Apply the Scharr filter for gy
                gy += scharr_y[m][n] * px_array[x_offset][y_offset]
        
        # Store the calculated gradient values in the edges arrays
        edges_x[i][j] = gx
        edges_y[i][j] = gy

    
    return edges_x, edges_y


# part 2)  Take the absolute value

def computeEdgeStrength(gx, gy):
    width, height = len(gx[0]), len(gx)
    edge_strength = createInitializedGreyscalePixelArray(width, height, initValue=0.0)
    
    # calculate the absolute sum
    for row in range(height):
        for col in range(width):
            edge_strength[row][col] = abs(gx[row][col]) + abs(gy[row][col])
    
    # Set border pixels to zero
    for i in range(height):
        edge_strength[i][0] = 0
        edge_strength[i][width-1] = 0
    for j in range(width):
        edge_strength[0][j] = 0
        edge_strength[height-1][j] = 0
    
    return edge_strength






# STEP 3) Apply 5x5 mean filter(s) to image. 


def applyMeanFilter(px_array):
    filter_size = 5
    padding = filter_size // 2
    width, height = len(px_array[0]), len(px_array)
    
    # Create an output image array initialized to 0.0
    filtered_array = createInitializedGreyscalePixelArray(width, height, initValue=0.0)
    
    # Iterate over each pixel, ignoring the boundary pixels
    for y in range(padding, height - padding):
        for x in range(padding, width - padding):
            # Collect the values from the 5x5 neighborhood
            neighborhood_sum = 0.0
            for ky in range(-padding, padding + 1):  # Kernel y-axis range
                for kx in range(-padding, padding + 1):  # Kernel x-axis range
                    neighborhood_sum += px_array[y + ky][x + kx]
            
            # Compute the mean of the neighborhood
            mean_value = neighborhood_sum / (filter_size * filter_size)
            
            # Set the computed mean value in the output array
            filtered_array[y][x] = abs(mean_value)
    
    # Set the outer boundary pixels to zero
    for x in range(width):
        filtered_array[0][x] = 0.0
        filtered_array[height - 1][x] = 0.0
    for y in range(height):
        filtered_array[y][0] = 0.0
        filtered_array[y][width - 1] = 0.0
    
    return filtered_array

# method that runs the mean filter multiple times
def applyMeanFilterMultipleTimes(px_array, times):
    result_array = px_array
    for _ in range(times):
        result_array = applyMeanFilter(result_array)
    return result_array






# STEP 4) Perform a simple thresholding operation

def computeThreshold(px_array):
    # Create an empty binary image array initialized with zeros
    width, height = len(px_array[0]), len(px_array)
    output = createInitializedGreyscalePixelArray(width, height)
    
    # Iterate through each pixel in the input image
    for i in range(height):
        for j in range(width):
            # Compare the pixel value with the threshold value
            if px_array[i][j] < 22:
                # If pixel value is smaller than the threshold, set corresponding pixel in binary image to 0
                output[i][j] = 0
            else:
                # If pixel value is greater than or equal to the threshold, set corresponding pixel in binary image to 255
                output[i][j] = 255
    
    # Return the binary image
    return output




# STEP 5) Perform several dilation steps followed by several erosion steps

# initialise the 5x5 kernal
kernel = [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0]
    ]
  
# dilate method
def dilate(px_array, kernel, iterations=1):
    # initialise variables
    height = len(px_array)
    width = len(px_array[0])
    kernel_size = len(kernel)
    padding = kernel_size // 2

    for _ in range(iterations):
        # Create a new array for the dilated image
        new_array = createInitializedGreyscalePixelArray(width, height)  
        for i in range(padding, height - padding):
            for j in range(padding, width - padding):
                max_val = 0
                for ki in range(-padding, padding + 1):
                    for kj in range(-padding, padding + 1):
                        # Check if the kernel position is active
                        if kernel[ki + padding][kj + padding] == 1:  
                            max_val = max(max_val, px_array[i + ki][j + kj]) 
                            # Set the pixel to the maximum value found
                new_array[i][j] = max_val  
        px_array = new_array  
    return px_array

# erode method
def erode(px_array, kernel, iterations=1):
    # initialise variables
    height = len(px_array)
    width = len(px_array[0])
    kernel_size = len(kernel)
    padding = kernel_size // 2

    for _ in range(iterations):
        # Create a new array for the eroded image
        new_array = createInitializedGreyscalePixelArray(width, height) 
        for i in range(padding, height - padding):
            for j in range(padding, width - padding):
                min_val = 255
                for ki in range(-padding, padding + 1):
                    for kj in range(-padding, padding + 1):
                        # Check if the kernel position is active
                        if kernel[ki + padding][kj + padding] == 1:  
                            min_val = min(min_val, px_array[i + ki][j + kj])  
                            # Set the pixel to the minimum value found
                new_array[i][j] = min_val  
        px_array = new_array 
    return px_array




# STEP 6) Perform a connected component analysis
def connectedComponentsLabeling(binary_image):
    height = len(binary_image)
    width = len(binary_image[0])
    
    # Initialize a labeled image with the same dimensions as the binary image, filled with 0s
    labeled_image = createInitializedGreyscalePixelArray(width, height)
    
    current_label = 1

    # Uniformity predicate function
    def is_similar(intensity1, intensity2, threshold=2):
        return abs(intensity1 - intensity2) <= threshold
    
    # Function to get 4-connected neighbors
    def get_neighbors(x, y):
        neighbors = []
        if x > 0:
            neighbors.append((x-1, y))
        if x < height - 1:
            neighbors.append((x+1, y))
        if y > 0:
            neighbors.append((x, y-1))
        if y < width - 1:
            neighbors.append((x, y+1))
        return neighbors
    
    for i in range(height):
        for j in range(width):
            if binary_image[i][j] == 255 and labeled_image[i][j] == 0:
                # Start a new region
                queue = deque([(i, j)])
                labeled_image[i][j] = current_label
                
                while queue:
                    x, y = queue.popleft()
                    for nx, ny in get_neighbors(x, y):
                        if binary_image[nx][ny] == 255 and labeled_image[nx][ny] == 0 and is_similar(binary_image[x][y], binary_image[nx][ny]):
                            labeled_image[nx][ny] = current_label
                            queue.append((nx, ny))
                
                current_label += 1
    
    return labeled_image


# STEP 7) Extract the bounding box(es) around all regions

def extractBoundingBoxes(labeled_image):
    height = len(labeled_image)
    width = len(labeled_image[0])
    
    # initialise bounding box 
    bounding_boxes = {}
    
    for i in range(height):
        for j in range(width):
            label = labeled_image[i][j]
            if label > 0:
                if label not in bounding_boxes:
                    # [min_x, min_y, max_x, max_y]
                    bounding_boxes[label] = [j, i, j, i]  
                else:
                    bounding_boxes[label][0] = min(bounding_boxes[label][0], j)
                    bounding_boxes[label][1] = min(bounding_boxes[label][1], i)
                    bounding_boxes[label][2] = max(bounding_boxes[label][2], j)
                    bounding_boxes[label][3] = max(bounding_boxes[label][3], i)
    
    # Convert bounding boxes to the required list format
    bounding_box_list = [box for box in bounding_boxes.values()]
    return bounding_box_list









# This is our code skeleton that performs the coin detection.
def main(input_path, output_path):
    # This is the default input image, you may change the 'image_name' variable to test other images.
    image_name = 'easy_case_6'
    input_filename = f'./Images/easy/{image_name}.png'
    if TEST_MODE:
        input_filename = input_path

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)
    
    # STEP 1) part 1: Convert RGB to grayscale
    px_array_grey = convertRGBToGreyscale(image_height,image_width,px_array_r, px_array_g, px_array_b)
    
    # STEP 1) part 2: Stretch grayscale values between 0 to 255 using 5-95 percentile strategy
    px_array_stretched = stretchGreyScaleValues(px_array_grey)
    
    fig, axs = pyplot.subplots(1, 1)
    axs.imshow(px_array_stretched, cmap='gray')
    pyplot.axis('off')
    pyplot.tight_layout()
    pyplot.savefig(f'./output_images/output_for_step1.png', bbox_inches='tight', pad_inches=0)

    # STEP 2) part 1: Apply a 3x3 Scharr filter in both directions
    edges_x, edges_y = applyScharrFilter(px_array_stretched)
    
    # STEP 2) part 2: Take the absolute value of the sum between horizontal (x) and vertical (y)
    edge_strength = computeEdgeStrength(edges_x, edges_y)
    
    fig, axs = pyplot.subplots(1, 1)
    axs.imshow(edge_strength, cmap='gray')
    pyplot.axis('off')
    pyplot.tight_layout()
    pyplot.savefig(f'./output_images/output_for_step2.png', bbox_inches='tight', pad_inches=0)
    
    # STEP 3) Apply 5x5 mean filter(s) to image. 
    blur = applyMeanFilterMultipleTimes(edge_strength,3)
    blurhist = calculateHistogram(blur)
    
    fig, axs = pyplot.subplots(1, 1)
    axs.imshow(blur, cmap='gray')
    pyplot.axis('off')
    pyplot.tight_layout()
    pyplot.savefig(f'./output_images/output_for_step3.png', bbox_inches='tight', pad_inches=0)
   
    # STEP 4) Perform a simple thresholding operation
    segmented = computeThreshold(blur)
    
    fig, axs = pyplot.subplots(1, 1)
    axs.imshow(segmented, cmap='gray')
    pyplot.axis('off')
    pyplot.tight_layout()
    pyplot.savefig(f'./output_images/output_for_step4.png', bbox_inches='tight', pad_inches=0)
    
    # STEP 5) Perform several dilation steps followed by several erosion steps
    dilated = dilate(segmented, kernel, iterations=7)
    eroded = erode(dilated, kernel, iterations=7)
    
    fig, axs = pyplot.subplots(1, 1)
    axs.imshow(eroded, cmap='gray')
    pyplot.axis('off')
    pyplot.tight_layout()
    pyplot.savefig(f'./output_images/output_for_step5.png', bbox_inches='tight', pad_inches=0)
    
    # STEP 6) Perform connected component analysis
    labeled_image = connectedComponentsLabeling(eroded)
    
    fig, axs = pyplot.subplots(1, 1)
    axs.imshow(labeled_image, cmap='gray')
    pyplot.axis('off')
    pyplot.tight_layout()
    pyplot.savefig(f'./output_images/output_for_step6.png', bbox_inches='tight', pad_inches=0)
    
    
    # STEP 7) Extract the bounding box(es) around all regions
    bounding_box_list = extractBoundingBoxes(labeled_image)
    
    fig, axs = pyplot.subplots(1, 1)
    axs.imshow(px_array_grey, cmap='gray')
    
     # Print bounding box list
    print("Bounding Box List:", bounding_box_list)
    
    # Draw bounding boxes
    for bounding_box in bounding_box_list:
        bbox_min_x = bounding_box[0]
        bbox_min_y = bounding_box[1]
        bbox_max_x = bounding_box[2]
        bbox_max_y = bounding_box[3]
        
        bbox_xy = (bbox_min_x, bbox_min_y)
        bbox_width = bbox_max_x - bbox_min_x
        bbox_height = bbox_max_y - bbox_min_y
        rect = Rectangle(bbox_xy, bbox_width, bbox_height, linewidth=2, edgecolor='r', facecolor='none')
        axs.add_patch(rect)
        
    pyplot.axis('off')
    pyplot.tight_layout()
    pyplot.savefig(f'./output_images/output_for_step7.png', bbox_inches='tight', pad_inches=0)
   

    
    
    #########################################
    # Bounding box coordinates information ###
    # bounding_box[0] = min x
    # bounding_box[1] = min y
    # bounding_box[2] = max x
    # bounding_box[3] = max y
    #########################################
    
    # bounding_box_list = [[74, 68, 312, 303]]  # This is a dummy bounding box list, please comment it out when testing your own code.
   
        

    # fig, (ax1, ax2, ax3, ax4) = pyplot.subplots(1, 4, figsize=(36, 6))
    # fig1,(ax5,ax6) = pyplot.subplots(1, 2, figsize=(10, 6))
    # fig2,(ax7) = pyplot.subplots(1, 1, figsize=(10, 6))
    # fig3, ax8 = pyplot.subplots(1, 1, figsize=(10, 6))
    # fig4, ax9 = pyplot.subplots(1, 1, figsize=(10, 6))
    
    # Display the stretched grayscale image
    # ax1.imshow(px_array_stretched, cmap='gray')
    # ax1.set_title("Stretched Grayscale Image")
    # ax1.axis('off')
    
    # # Display the horizontal edge map
    # ax2.imshow(edges_x, cmap='gray')
    # ax2.set_title("Edge map (gx) on horizontal direction")
    # ax2.axis('off')

    # # Display the vertical edge map
    # ax3.imshow(edges_y, cmap='gray')
    # ax3.set_title("Edge map (gy) on vertical direction")
    # ax3.axis('off')
    
    #  # Display the edge strength map
    # ax4.imshow(edge_strength, cmap='gray')
    # ax4.set_title("Edge Strength (gm)")
    # ax4.axis('off')
    
    # # Display the mean filtered image
    # ax5.imshow(blur, cmap='gray')
    # ax5.set_title("Mean Filtered Image")
    # ax5.axis('off')
    
    # # Display the histogram
    # ax6.bar(range(256), blurhist)
    # ax6.set_title("Grayscale Histogram")
    # ax6.set_xlim(0, 255)
    
    # # Display segmented
    # ax7.imshow(segmented,cmap='gray')
    # ax7.set_title("segmented")
    # ax7.axis('off')
    
     # Display the result after dilation and erosion
    # ax8.imshow(eroded, cmap='gray')
    # ax8.set_title("Dilated and Eroded Image")
    # ax8.axis('off')
    
    # # Display the labeled image
    # ax9.imshow(labeled_image, cmap='nipy_spectral')
    # ax9.set_title("Connected Components")
    # ax9.axis('off')


  # Draw bounding boxes
    # for bounding_box in bounding_box_list:
    #     bbox_min_x = bounding_box[0]
    #     bbox_min_y = bounding_box[1]
    #     bbox_max_x = bounding_box[2]
    #     bbox_max_y = bounding_box[3]
        
    #     bbox_xy = (bbox_min_x, bbox_min_y)
    #     bbox_width = bbox_max_x - bbox_min_x
    #     bbox_height = bbox_max_y - bbox_min_y
    #     rect = Rectangle(bbox_xy, bbox_width, bbox_height, linewidth=2, edgecolor='r', facecolor='none')
    #     ax9.add_patch(rect)
    


    pyplot.tight_layout()
    default_output_path = f'./output_images/{image_name}_edges.png'
    if not TEST_MODE:
        pyplot.savefig(default_output_path, bbox_inches='tight', pad_inches=0)
        pyplot.show()
    else:
        pyplot.savefig(output_path, bbox_inches='tight', pad_inches=0)



if __name__ == "__main__":
    num_of_args = len(sys.argv) - 1
    
    input_path = None
    output_path = None
    if num_of_args > 0:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        TEST_MODE = True
    
    main(input_path, output_path)
    