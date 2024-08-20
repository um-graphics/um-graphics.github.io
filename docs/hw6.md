---
layout: spec
permalink: /hw6
latex: true

title: Homework 6 – 3D Deep Learning
due: 11:59 p.m. on Wednesday April 17th, 2024
---

<link href="style.css" rel="stylesheet">
<div style="display:none">
    <!-- Define LaTeX commands here -->
    \(
		\newcommand{\EB}{\mathbf{E}}
		\newcommand{\FB}{\mathbf{F}}
		\newcommand{\IB}{\mathbf{I}}
		\newcommand{\KB}{\mathbf{K}}
        \newcommand{\MB}{\mathbf{M}}
		\newcommand{\RB}{\mathbf{R}}
		\newcommand{\XB}{\mathbf{X}}
		\newcommand{\pB}{\mathbf{p}}
		\newcommand{\tB}{\mathbf{t}}
		\newcommand{\zeroB}{\mathbf{0}}
    \)
</div>

{% capture code %}<i class="fa fa-code icon-large"></i>{% endcapture %}
{% capture autograde %}<i class="fa fa-robot icon-large"></i>{% endcapture %}
{% capture report %}<i class="fa fa-file icon-large"></i>{% endcapture %}

# Homework 6 – 3D Deep Learning

## Instructions

This homework is **due at {{ page.due }}**.

The submission includes two parts:
1. **To Canvas**: submit a `zip` file containing a **single** directory with your **uniqname** as the name that contains all your code and anything else asked for on the [Canvas Submission Checklist](#canvas-submission-checklist). Don't add unnecessary files or directories.

    {{ code }} -
   <span class="code">We have indicated questions where you have to do something in code in red. **If Gradescope asks for it, also submit your code in the report with the formatting below. Please include the code in your gradescope submission.** </span> 

    Starter code is given to you on Canvas under the "Homework 6" assignment. You can also download it [here](https://drive.google.com/file/d/1z2UEf4goLwt5dTjVToXOmjSehIcQrRNB/view?usp=sharing). Clean up your submission to include only the necessary files. Pay close attention to filenames for autograding purposes. 
        
    <div class="primer-spec-callout info" markdown="1">
      **Submission Tip:** Use the [Tasks Checklist](#tasks-checklist) and [Canvas Submission Checklist](#canvas-submission-checklist) at the end of this homework. 

2. **To Gradescope**: submit a `pdf` file as your write-up, including your answers to all the questions and key choices you made.

    {{ report }} -
   <span class="report">We have indicated questions where you have to do something in the report in green. **For coding questions please include in the code in the report.**</span>

    The write-up must be an electronic version. **No handwriting, including plotting questions.** $$\LaTeX$$ is recommended but not mandatory.

   For including code, **do not use screenshots**. Generate a PDF using a [tool like this](https://www.i2pdf.com/source-code-to-pdf){:target="_blank"} or using this [Overleaf LaTeX template](https://www.overleaf.com/read/wbpyympmgfkf#bac472){:target="_blank"}. If this PDF contains only code, be sure to append it to the end of your report and match the questions carefully on Gradescope.

**NERF**:
- You'll be writing the code in the same way you've been writing for Homework 4 Part 2, i.e., [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true){:target="_blank"}. You may use local [Jupyter Notebooks](https://jupyter.org/){:target="_blank"}, however suggest that you use Google Colab, as running these models on local system may consume too much of GPU RAM (This assignment is not **CPU Friendly** like Homework 4.).

- We suggest you follow these steps to setup the codebase for this assignment depending on whether you are using Google Colab or your local system.

    **Google Colab**: Steps to Setup the Codebase

    1. Download and extract the zip file. 
    2. Upload the folder containing the entire code (with the notebook) to your Google Drive. 
    3. Ensure that you are using the GPU session by using the `Runtime -> Change Runtime Type` and selecting `Python3` and `T4 GPU`. Start the session by clicking `Connect` at the top right. The default T4 GPU should suffice for you to complete this assignment.

    **Local System**: Steps to Setup the Codebase 
    1. Download and extract the zip file to your local directory.
	2. Get rid of the command lines that is specfic for google colabs.
    3. You are good to start Section 2 of the assignment.


### Python Environment

The autograder uses Python 3.7. Consider referring to the [Python standard library docs](https://docs.python.org/3.7/library/index.html){:target="_blank"} when you have questions about Python utilties.

To make your life easier, we recommend you to install the latest [Anaconda](https://www.anaconda.com/download/){:target="_blank"} for Python 3.7. This is a Python package manager that includes most of the modules you need for this course. We will make use of the following packages extensively in this course:
- [Numpy](https://numpy.org/doc/stable/user/quickstart.html){:target="_blank"}
- [Matplotlib](https://matplotlib.org/stable/tutorials/introductory/pyplot.html){:target="_blank"}
- [OpenCV](https://opencv.org/){:target="_blank"}

<!-- ## Camera Calibration -->

<figure class="figure-container">
    <div class="flex-container">
        <figure>
            <img src="{{site.url}}/assets/hw5/figures/temple_us.png" alt="Temple" width="300">
            <figcaption>Temple</figcaption>
        </figure>
        <figure>
            <img src="{{site.url}}/assets/hw5/figures/zrtrans_us.png" alt="zrtrans" width="300">
            <figcaption>zrtrans</figcaption>
        </figure>
        <figure>
            <img src="{{site.url}}/assets/hw5/figures/reallyInwards_us.png" alt="reallyInwards" width="300">
            <figcaption>reallyInwards</figcaption>
        </figure>
    </div>
    <figcaption>Figure 1: Epipolar lines for some of the datasets</figcaption>
</figure>

<!-- 
### Task 1: Estimating $$\MB$$

We will give you a set of 3D points $$\{\XB_i\}_i$$ and corresponding 2D points $$\{\pB_i\}_i$$. In part 1, you're given corresponding point locations in `pts2d-norm-pic.txt` and `pts3d-norm.txt`, which corresponds to a camera projection matrix. **Solve** the projection matrix $$P$$ and **include** it in your report. The goal is to compute the projection matrix $$\MB$$ that maps from world 3D coordinates to 2D image coordinates. Recall that 

$$
\pB \equiv \MB \XB
$$ 

and by deriving an optimization problem. The script `task1.py` shows you how to load the data. The data we want you to use is in `task1/`, but we show you how to use data from Task 2 and 3 as well. **Credit:** The data from task 1 and an early version of the problem comes from James Hays's Georgia Tech CS 6476. 

1.  *(15 points)* {{ code }} <span class="code">Fill in `find_projection` in `task1.py`.</span>

2.  *(5 points)* {{ report }} <span class="report">Report $$\MB$$</span> for the data in `task1/`.

3.  *(10 points)* {{ code }} <span class="code">Fill in `compute_distance` in `task1.py`.</span> 
	
	In this question, you need to compute the average distance in the image plane (i.e., pixel locations) between the homogeneous points $$\MB \XB_i$$ and 2D image coordinates $$\pB_i$$, or

	$$
	\frac{1}{N} \sum_{i}^{N} ||\textrm{proj}(\MB\XB_i) - \pB_i||_2 .
	$$

	where $$\textrm{proj}([x,y,w]) = [x/w, y/w]$$.
	The distance quantifies how well the projection maps the points $$\XB_i$$ to $$\pB_i$$. You should use `find_projection` from earlier.
	
	Note: You should feel good about the distance if it is **less than 0.01** for the given sample data. If you plug in different data, this threshold will of course vary. -->


## Estimation of the Fundamental Matrix and Epipoles

**Data:** we give you a series of datasets that are nicely bundled in the folder `task1/`. Each dataset contains two images `img1.png` and `img2.png` and a numpy file `data.npz` containing a whole bunch of variables. The script `task1.py` shows how to load the data.

**Credit:** `temple` comes from Middlebury's Multiview Stereo dataset.


### Task 1: Estimating $$\FB$$ and Epipoles

1.  *(15 points)* {{ code }} <span class="code">Fill in `find_fundamental_matrix`</span> in `task1.py`. You should implement the eight-point algorithm mentioned in the lecture. Remember to normalize the data  and to reduce the rank of $$\FB$$. For normalization,
you can scale the image size and center the data at 0. We want you to "estimate" the fundamental matrix here so it's ok for your result to be slighly off from the opencv implementation. 

2.  *(10 points)* {{ code }} <span class="code">Fill in `compute_epipoles`.</span> This should return the homogeneous coordinates of the epipoles -- remember they can be infinitely far away! For computing the nullspace of F, using SVD would be helpful!

3.  *(5 points)* {{ report }} <span class="report">Show epipolar lines for `temple`, `reallyInwards`, and another dataset of your choice.</span>

4.  *(5 points)* {{ report }} <span class="report">Report the epipoles for `reallyInwards` and `xtrans`</span>.

<!-- 
### Task 3: Triangulating $$\XB$$

<figure class="figure-container">
  <img src="{{site.url}}/assets/hw5/figures/reallyInwards_rec.png" alt="reallyInwards_rec" width="500">
  <figcaption>Figure 2: Visualizations of reallyInwards reconstructions</figcaption>
</figure>

The next step is extracting 3D points from 2D points and camera matrices, which is called triangulation. Let $$\XB$$ be a point in 3D.

$$
\pB = \MB_1 \XB ~~~~ \pB' = \MB_2 \XB
$$

Triangulation solves for $$\XB$$ given $$\pB, \pB', \MB_1, \MB_2$$. We'll use OpenCV's algorithms to do this.

1.  *(5 points)* {{ report }} <span class="report">Compute the Essential Matrix $$\EB$$ for the Fundamental Matrix $$\FB$$.</span> You should do this for the dataset `reallyInwards`. Recall that

	$$
	\FB = \KB'^{-T} \EB \KB^{-1}
	$$

	and that $$\KB, \KB'$$ are always invertible (for reasonable cameras), so you can compute $$\EB$$ straightforwardly.

2.  *(15 points)* {{ code }} <span class="code">Fill in `find_triangulation` in `task23.py`.</span> 

	The first camera's projection matrix is $$\KB[\IB,\zeroB]$$. The second camera's projection matrix can be obtained by decomposing $$\EB$$ into a rotation and translation via `cv2.decomposeEssentialMat`. (Note: $$\EB$$ can be obtained using the formula from part a) This function returns two matrices $$\RB_1$$ and $$\RB_2$$ and a translation $$\tB$$. The four possible camera matrices for $$\MB_2$$ are: 	
	
	$$ 	
	\MB_2^1 = \KB' [\RB_1, \tB],~~~~\MB_2^2 = \KB' [\RB_1, -\tB],~~~~\MB_2^3 = \KB' [\RB_2, \tB],~~~~\MB_2^4 = \KB' [\RB_2, -\tB] 	
	$$
	
	You can identify which projection is correct by picking the one for which the most 3D points are in front 	of both cameras. This can be done by checking for the positive depth, which can be done by looking at the last entry of the homogeneous coordinate: the extrinsics put the 3D point in the camera's frame, where $$z<0$$ is behind the camera, and the last row of $$\KB$$ is $$[0,0,1]$$ so this does not change things.

	Finally, triangulate the 2D points using `cv2.triangulatePoints`.

3.  *(10 points)* {{ report }} <span class="report">Put a visualization of the point cloud for `reallyInwards` in your report.</span> You can use `visualize_pcd` in `utils.py` or implement your own. -->

## 3D Generation

### Task 2: Neural radiance fields 

We will fit a neural radiance field (NeRF) to a collection of photos (with their camera pose), and use it to render a scene from different (previously unseen) viewpoints. To estimate the color of a pixel, we will estimate the 3D ray that exist the pixel. Then, we will walk in the direction of the ray and query the network at each point. Finally, we will use volume rendering to obtain the pixel’s RGB color, thereby accounting for occlusion.

It is an MLP 
$$F_\Theta$$ 
such that

$$
F_\Theta(x, y, z, \theta, \phi) = (R, G, B, \sigma)
$$

where 
$$(x, y, z)$$ 
is a 3D point in the scene, and 
$$(\theta, \phi)$$
is a viewing direction. It returns a color 
$$(R, G, B)$$ 
and a (non-negative) density 
$$( \sigma)$$ 
that indicates whether this point in space is occupied.

1.  *(10 points)* {{ code }} <span class="code"> Implement the function positional_encoder(x, L_embed = 6) </span> that
encodes the input x as 
$$\gamma(x) = (x, \sin(2^{0}x), \cos(2^{0}x), \ldots, \sin(2^{L_{embed}-1}x), \cos(2^{L_{embed}-1}x)).$$  

2.  *(10 points)* {{ code }} <span class="code"> Implement the code that samples 3D points along a ray in `render`.</span>  This will
be used to march along the ray and query 
$$F_\Theta$$  

3.  *(10 points)* {{ code }} <span class="code"> After having walked along the ray and queried 
$$F_\Theta$$ 
at each point in `render`, we will estimate the pixel's color, represented as rgb_map. </span>  We will also compute, depth_map, which indicates the depth of the nearest surface at this pixel. 

4.  *(10 points)* {{ code }} <span class="code"> Please implement part of the `train(model, optimizer, n_iters)` function. </span> 
In the training loop, the model is trained to fit one image randomly picked from the dataset at each iteration.
You need to tune the near and far point parameter in get rays to make maximize the clarity of the RGB prediction image

5.   *(5 points)* {{ report }} <span class="report">Please include the best picture (after parameter tuning) of your RGB prediction, depth prediction, and groud truth figure for different view points. </span> 


We can now render the NeRF from different viewpoints. The predicted image should be pretty similar to the ground truth but it may have less clarity.

# Tasks Checklist

This section is meant to help you keep track of the many things that go in the report:

- [ ] **Estimating $$F$$ and Epipoles**
	- [ ] 1.1 - {{ code }} `find_fundamental_matrix`
	- [ ] 1.2 - {{ code }} `compute_epipoles`
	- [ ] 1.3 - {{ report }} Epipolar lines for `temple`, `reallyInwards`, and your choice
	- [ ] 1.4 - {{ report }} Epipoles for `reallyInwards` and `xtrans`

- [ ] **Neural radiance fields**
    - [ ] 2.1 - {{ code }} Implement `positional_encoder`
    - [ ] 2.2 - {{ code }} Sample 3D points along a ray
    - [ ] 2.3 - {{ code }} Estimate `rgb_map` and `depth_map`
    - [ ] 2.4 - {{ code }} `train`
	- [ ] 2.5 - {{ report }} Report your prediction and groud truth image

# Canvas Submission Checklist

In the `zip` file you submit to Canvas, the directory named after your uniqname should include the following files:
- [ ] `task1.py`
- [ ] `HW6_Neural_Radiance_Fields.ipynb`
