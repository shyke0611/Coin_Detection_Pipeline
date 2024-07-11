<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->

[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:hyungkenine2003@gmail.com)
[![LinkedIn][linkedin-shield]][linkedin-url]
[![Website](https://img.shields.io/badge/Website-Visit-blue?style=for-the-badge)](https://andrewshinportfolio.netlify.app)


<!-- PROJECT LOGO -->
<br />
<div align="center">

  <h3 align="center">Coin Detection Pipeline Project</h3>

  <p align="center">
    Software Engineering part 3 assignment for The University of Auckland, handling Image Processing
  </p>
</div>

<br><br>

<!-- ABOUT THE PROJECT -->
## About The Project
[Assignment Overview.pdf](https://github.com/user-attachments/files/16166802/CS373_coin_detection_assignment.pdf)

This project involves developing a Python pipeline to detect and outline coins in images using techniques such as greyscale conversion, edge detection, blurring, thresholding, morphological operations, connected component analysis, and drawing bounding boxes. An extension includes advanced methods for harder images and coin identification.


[![full pipeline][full pipeline]](project_images/fullpipeline.png)
<br><br>

[![my pipeline][my pipeline]](project_images/mypipeline.png)
<br><br>

[![terminal][terminal]](project_images/terminal.png)

## Project Steps

### 1. Convert to Greyscale and Normalize
- **Convert to Greyscale**: Convert the RGB image to greyscale using the ratio 0.3 * Red + 0.6 * Green + 0.1 * Blue, and round the pixel values to the nearest integer.
- **Contrast Stretching**: Stretch the values between 0 to 255 using the 5-95 percentile strategy.

### 2. Edge Detection
- **Scharr Filter**: Apply a 3x3 Scharr filter in horizontal and vertical directions to get edge maps.
- **Edge Strength**: Compute the absolute value of the sum of horizontal and vertical edge maps.

### 3. Image Blurring
- **Mean Filter**: Apply a 5x5 mean filter to the image, taking the absolute value after computing each window and applying the filter three times sequentially.

### 4. Threshold the Image
- **Thresholding**: Perform a simple thresholding operation to segment the coins from the background, resulting in a binary image.

### 5. Erosion and Dilation
- **Morphological Operations**: Perform several dilation steps followed by erosion steps using a circular 5x5 kernel to refine the binary image.

### 6. Connected Component Analysis
- **Component Analysis**: Perform connected component analysis to find all connected components in the binary image.

### 7. Draw Bounding Box
- **Bounding Boxes**: Extract bounding boxes around all detected coin regions by identifying the minimum and maximum x and y coordinates of the pixels in each connected component.

### Additional Steps
[Additional_steps_report.pdf](https://github.com/user-attachments/files/16061014/373.assignment.extension.report.hshi270.pdf)

- **Laplacian Filter**: Use the Laplacian filter for edge detection.
- **Hard-Level Images**: Test the pipeline on harder images and possibly identify coin types based on size.
- **Coin Counting and Identification**: Output the number of detected coins and identify their types (e.g., 1-dollar coin, 50-cent coin) based on their size.


## Built With

* [Python]


## Prerequisites

Before running the project, ensure you have the following installed:

1. **Python**: Ensure you have Python 3 installed. You can download it from [python.org](https://www.python.org/).
2. **Matplotlib**: The only external library allowed for the main component is Matplotlib. You can install it using pip:

   ```sh
   pip install matplotlib

     
<!-- REFLECTION -->
## Reflection

#### Context and Motivation
This Coin Detection Assignment was an intensive project aimed at enhancing my Python programming skills and understanding of image processing techniques. I embarked on this project to deepen my knowledge in computer vision and its practical applications. It provided an excellent opportunity to expand my expertise in technical aspects as well as in designing effective algorithms.

#### Project Goals
The primary goal was to develop a comprehensive pipeline for detecting and outlining coins in images. The project involved implementing various image processing steps, including greyscale conversion, edge detection, image blurring, thresholding, morphological operations, and connected component analysis. An additional objective was to extend the pipeline to handle more challenging images and identify different types of coins based on size.

#### Challenges and Learning Experience
A significant challenge was ensuring the accuracy and robustness of the coin detection pipeline across different image complexities. This required extensive experimentation with various image processing techniques and fine-tuning parameters. Addressing these issues improved my understanding of computer vision concepts and enhanced my problem-solving abilities. Moreover, I learnt the importance of usability, memorability, and emotional impact in designing a user-friendly system. Integrating Nielsen's heuristics and principles of design such as unity, balance, and emphasis into the pipeline was crucial in achieving a seamless user experience.


<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/andrew-hk-shin
[my pipeline]: project_images/mypipeline.png
[terminal]: project_images/terminal.png
[full pipeline]: project_images/fullpipeline.png
