---
layout: spec
---

<link href="style.css" rel="stylesheet">

# Overview

- **Instructor**: Jeong Joon Park
- **GSIs**: 
	- Alex Janosi
	- Anurekha Ravikumar
	- Farzad Siraj
	- Jinfan Zhou
	- Shrikant Arvavasu
- **IA**: 
	- Yuhang Ning (dlning)
- **Lecture**: Monday/Wednesday 10:30 AM - 12:00 Noon, STAMPS
- **Discussions**: 
	- Monday 12:30-1:30PM, 2166 DOW - Alex
	- Monday 3:30-4:30PM, 1005 DOW - Anurekha
	- Wendesday 3:30-4:30PM, 3150 DOW - Farzad
	- Wednesday 4:30-5:30PM, 3150 DOW - Yuhang [Zoom Link](https://umich.zoom.us/j/3645727237){:target="_blank"}
	- Wednesday 5:30-6:30PM, 107 GFL - Jinfan
	- Thursday 3:30-4:30PM, 3150 DOW - Shrikant
- [**Piazza Signup**](https://piazza.com/umich/winter2024/c9b2){:target="_blank"}
- [**Lecture Recordings**](https://leccap.engin.umich.edu/leccap/site/rxk3s2yc2cg9pggod0u){:target="_blank"} (It could take a couple of days to be processed and uploaded)
- [**Course Calendar**](https://calendar.google.com/calendar/u/0?cid=Y18zYjdmOTM1ODhjOTk2NDg0YTM5OTRkMTc1NWIwZWM5MzViZWZlMjgzYzI2ZjA1NjlkNGEzNDE5ZWI5M2ZlZmM1QGdyb3VwLmNhbGVuZGFyLmdvb2dsZS5jb20){:target="_blank"}
- [**Virtual Office Hours**](https://officehours.it.umich.edu/queue/1511){:target="_blank"}
- Course email: eecs442-wn24@umich.edu
- Note on Waitlist: We DO NOT reorder waitlist. Please talk to the undergraduate advising office regarding your course enrollment.

## Homeworks
- [Homework 1: Numbers and Images](hw1.md)
- [Homework 2: Convolution and Feature Detection](hw2.md)
- [Homework 3: Fitting Models and Image Warping](hw3.md)
- [Homework 4: Machine Learning](hw4.md)
- [Homework 5: Generative Models](hw5.md)
- [Homework 6: 3D Vision](hw6.md)

# Schedule
Tentative Schedule, details are subject to change. Refer to [Textbooks]({{ site.url }}/#textbooks) for textbook acronyms in readings.

| Date                  | Topic                                                                                       | Material |
|-----------------------|:-------------------------------------------------------------------------------------------:|----------|
| Wednesday<br>Jan 10   | **Introduction + Cameras 1**<br>Overview, Logistics, Pinhole Model, Homogeneous Coordinates | [Slides](https://drive.google.com/file/d/1aXUBzyCN9x5M7lZ7u5ah_pDzalEVuKM3/view?usp=sharing){:target="_blank"} <br> Reading: S2.1, H&Z 2, 6|
| Monday<br>Jan 15      | **No Class**<br>Martin Luther King Day                                                      |          |
| Wednesday<br>Jan 17   | **Cameras 2**<br>Intrinsics & Extrinsic Matrices, Lenses<br> :warning: ***Homework 1 Release***       | [Slides](https://drive.google.com/file/d/1ZXgW8aW4eFg_0HmUg2gtdBQsenlwuBRf/view?usp=sharing){:target="_blank"} <br> [Discussion Slides](https://drive.google.com/file/d/1r6LT5Yb0YDFD_OsYaviFuPz-1ogBxhSb/view?usp=sharing){:target="_blank"} <br> Reading: S2.1, H&Z 2, 6  |
| Monday<br>Jan 22      | **Math Recap**<br>Floating point numbers, Linear Algebra, Calculus                          | [Slides](https://drive.google.com/file/d/13b1OKtvBM6DjC0YFHWL79TOeWJqFj94t/view?usp=sharing) <br> Reading: Kolter |
| Wednesday<br>Jan 24   | **Light & Shading**<br>Human Vision, Color Vision, Reflection                               | [Slides](https://drive.google.com/file/d/1IfhiQVtWPvEWAJc-6JiIOeOF79MIitQD/view?usp=drive_link)   |
| Monday<br>Jan 29      | **Filtering**<br>Linear Filters, Blurring, Separable Filters, Gradients                     | [Slides](https://drive.google.com/file/d/1w9PYKGMQFZOUcsRin2OiQLuAjEQ2hoiq/view?usp=drive_link) <br> [Discussion Slides](https://drive.google.com/file/d/10VO-aPQcS9DEdsTUFyF5QHrY3SPWJHEZ/view?usp=sharing){:target="_blank"} <br> Reading: S2.2, S2.3  |
| Wednesday<br>Jan 31   | **Detectors & Discriptors 1**<br>Edge Detection, Gaussian Derivatives, Harris Corners<br> :warning: ***Homework 1 Due*** <br> :warning: ***Homework 2 Release***      | [Slides](https://drive.google.com/file/d/1tbWzD0ZV3GVpXNOU3bapyBAs_38I1mG4/view?usp=drive_link)   |
| Monday<br>Feb 5       | **Detectors & Discriptors**<br>Scale-Space, Laplacian Blob Detection, SIFT                  | [Slides](https://drive.google.com/file/d/1E2akIP5JJh6isYe0fc0XHUTi9fK-Y3aA/view?usp=sharing)  <br> [Discussion Slides](https://drive.google.com/file/d/1tvrcA1VE4fPynXv4S0C0wllhh7QDj4m1/view?usp=sharing){:target="_blank"} |
| Wednesday<br>Feb 7    | **Transforms 1**<br>Linear Regression, Total Least Squares, RANSAC, Hough Transform         | [Slides](https://drive.google.com/file/d/1P53vKodULvItUZsiofM7gX-n9w6ypHe2/view?usp=drive_link) <br> Reading: S2.1, S6  |
| Monday<br>Feb 12      | **Transforms 2**<br>Affine and Perspective Transforms, Fitting Transformations              | [Slides](https://drive.google.com/file/d/10zCinBKbwVRUf1xYguH7xQt8vkPzSR7x/view?usp=sharing) <br> Reading: S2.1, S6 |
| Wednesday<br>Feb 14   | **Machine Learning**<br>Supervised Learning, Linear Regression, Regularization <br> :warning: ***Homework 2 Due*** <br> :warning: ***Homework 3 Release*** | [Slides](https://drive.google.com/file/d/1uWj2ExVESlJk-B1vrQIhBm6eby5OojhF/view?usp=sharing) <br> Reading: ESL 3.1, 3.2(skim) |
| Monday<br>Feb 19      | **Optimization**<br>SGD, SGD+Momentum                                                       | [Slides](https://drive.google.com/file/d/1jIbF9FmzbOLBdeyJYaJPNZTfa-UmQzd0/view?usp=sharing) <br> [Discussion Slides](https://drive.google.com/file/d/19ipGNV8il5sr8EmCdmgmK5j8odIqy8XG/view?usp=sharing){:target="_blank"} <br> [HW3 Help Sheet](https://drive.google.com/file/d/1a7PcqSPGOrVgQXI7hegkMCGBOZk_Fu2Z/view?usp=sharing){:target="_blank"} |
| Wednesday<br>Feb 21   | **Neural Networks**<br>Backpropagation, Fully Connected Neural Networks                     | [Slides](https://drive.google.com/file/d/1yCVPywX8TdDhJ36as9cni1R7LV-2NMb_/view?usp=sharing)   |
| Monday<br>Feb 26      | **No Class**<br> :sunny: :beach_umbrella: Spring Break                                      |          |
| Wednesday<br>Feb 28   | **No Class**<br> :sunny: :beach_umbrella: Spring Break                                      |          |
| Monday<br>Mar 4       | **Convolutional Networks 1**<br>Convolution, Pooling                                        | [Slides](https://drive.google.com/file/d/1NuiWs8IekrWrA0WSrGLOo47GXVFRkWwI/view?usp=sharing)  <br> [Discussion Slides](https://drive.google.com/file/d/1F6Rpd4Rxv71A84mvN3FV6krmNmUTVQIz/view?usp=sharing){:target="_blank"} |
| Wednesday<br>Mar 6    | **Convolutional Networks 2**<br>CNN Architectures, Training Methods & Techniques <br> :warning: ***Homework 3 Due*** <br> :warning: ***Homework 4 Release*** | [Slides](https://drive.google.com/file/d/1cyf2MKjxIpV0O2oX248xsjMUklxK8FuA/view?usp=sharing) <br> |
| Monday<br>Mar 11      | **Segmentation**<br>Semantic/Instance Segmentation                                          | [Slides](https://drive.google.com/file/d/1WLyMTrLYP9NWOix3TNrs782wULaMBCDf/view?usp=sharing) <br> [Discussion Slides](https://drive.google.com/file/d/1oUApPFzrnCqkMhOczp7Bdj7D7bYSuTZT/view?usp=sharing){:target="_blank"}   |
| Wednesday<br>Mar 13   | **Detection & Other Topics**                                                                | [Slides](https://drive.google.com/file/d/1rprE23d8BLKxDhiWgyhc4ak7ZNC0dN0b/view?usp=sharing)   |
| Monday<br>Mar 18      | **Image Generative Models 1** <br> Generative models, GANs, Self-supervised learning                                              | [Slides](https://drive.google.com/file/d/1AyGUz8Q4nJtER9FRN_XJJHClkpDYRLPj/view?usp=sharing) |
| Wednesday<br>Mar 20   | **Image Generative Models 2** <br> Score-based Models, Diffusion Models <br> :warning: ***Homework 4 Due*** <br> :warning: ***Homework 5 Release*** | [Slides](https://drive.google.com/file/d/1knipLfQXMBZjR8r5LRx5BvRO03mAq8N1/view?usp=sharing) |
| Monday<br>Mar 25      | :exclamation: **Midterm**                                                                   |  [Discussion Slides](https://drive.google.com/file/d/1TWnO10qeycd9Xw7vl12yPKSO-P07htMi/view?usp=sharing)    |
| Wednesday<br>Mar 27   | **Camera Calibration**<br>Intro to 3D, Camera Calibration <br> ***Project Proposal Due***                                  | [Slides](https://drive.google.com/file/d/14PCcRXUSY3ESZB_rB9QYoAlAyClU0UIx/view?usp=sharing)  <br> Reading: S6.3 |
| Monday<br>April 1     | **Epipolar Geometry**<br>Epipolar Geometry, The Fundamental & Essential Matrices            | [Slides](https://drive.google.com/file/d/1QMw75DV2tsDEXhi1cNWwT87jAx5ZIWz4/view?usp=sharing)  <br> Reading: S11 |
| Wednesday<br>April 3  | **Stereo**<br>Two-view Stereo <br> :warning: ***Homework 5 Due*** <br> :warning: ***Homework 6 Release*** | [Slides](https://drive.google.com/file/d/1hLkIzC7kN9nea6CblUphKv6EEL27YLmC/view?usp=sharing)  <br> [Discussion Slides](https://drive.google.com/file/d/1vBY3YCdSDtK2nv7qIWxIo8pHljaeWj-P/view?usp=sharing){:target="_blank"}  |
| Monday<br>April 8     | **Structure from Motion**<br>                        | [Slides](https://drive.google.com/file/d/19AJW_54JGIekWB4ptmGJY0NiPQiBA_Na/view?usp=drive_link)  <br> Reading: S7 |
| Wednesday<br>April 10 | **Neural Fields 1**<br> 3D Representations, Neural 3D reconstruction                                                                        | [Slides](https://drive.google.com/file/d/1F6WPtUAimcS2jz4AfBntY8SK7LEapEyq/view?usp=sharing)   |
| Monday<br>April 15    | **Neural Fields 2**                                                                           | [Slides](https://drive.google.com/file/d/1KyNh0PVR0aznO05CFv6NFBYL9bspFRpl/view?usp=sharing)   |
| Wednesday<br>April 17 | **Special Topics (Guest Lecture)** <br> :warning: ***Homework 6 Due***                                   | Slides   |
| Monday<br>April 22    | **Special Topics**<br> Transformers, Ethics                                                                          | [Slides](https://drive.google.com/file/d/1Pf8plwtdCsZBNBo-ZXoWtl50VSI1Q-fX/view?usp=sharing)   |
| Saturday<br>April 29  | **Project Report Due**                                                                      | Slides   |

# Syllabus

## Prerequisites

Concretely, we will assume that you are familiar with the following topics and will not review them in class:
- **Programming** - Algorithms and Data Structures at the level of EECS 281.
- **Python** - All course assignments will involve programming in Python.

It would be helpful for you to have a background in these topics. We will provide refreshers on these topics, but we will not go through a comprehensive treatment:
- **Array Manipulation** - Homework assignments will extensively involve manipulating multidimensional arrays with [NumPy](https://numpy.org/){:target="_blank"} and [PyTorch](https://pytorch.org/){:target="_blank"}. Some prior exposure will be useful, but if you've never used them before, then the first homework assignment will help you get up to speed.
- **Linear Algebra** - In addition to basic matrix and vector operations, you will need to know about the cross product, eigenvectors, and singular value decomposition.
- **Calculus** - You should be comfortable with the chain rule, and taking partial derivatives of vector-valued functions.

Much of computer vision is applying linear algebra to real-world data. If you are unfamiliar with linear algebra or calculus, past experience suggests that you are likely to struggle with the course. If you are rusty, we will provide math refreshers on the necessary topics, however, they are not meant as a first introduction.

## Textbooks

There is no required textbook. Particularly thorny homeworks will often come with lecture notes to help. The following optional books may be useful, and we will provide suggested reading from these books to accompany some lectures:
- Computer Vision: Algorithms and Applications by Richard Szeliski: [Available for free online here](http://szeliski.org/Book/){:target="_blank"}. (S)
- Computer vision: A Modern Approach (Second Edition), by David Forsyth and Jean Ponce. 
- Elements of Statistical Learning by Trevor Hastie, Robert Tibshirani, and Jerome Friedman. [Available for free online here](https://hastie.su.domains/ElemStatLearn/){:target="_blank"}. (ESL)
- Multiple View Geometry in Computer Vision (Second Edition), by Richard Hartley and Andrew Zisserman. [Available for free online through the UM Library (login required)](https://ebookcentral.proquest.com/lib/umichigan/detail.action?docID=256634){:target="_blank"}. (H&Z)
- Linear Algebra review and reference, by Zico Kolter. [Available for free online here](https://www.cs.cmu.edu/~zkolter/course/15-884/linalg-review.pdf){:target="_blank"}. (Kotler)

## Grading

Your grade will be based on:
- **Homework (60%)**: There will be six homeworks over the course of the semester. Each is worth 10%.
- **Midterm (20%)**: There will be a midterm in-class.
- **Final Project (20%)**: There will be a final project, in which you work in groups of 3-4 students to produce a substantial course project over the second half of the semester. This will consist of a proposal (worth 2%), and final report and video (worth 18%).

## Project Guidelines
See [here](proj.md) for details.

## Contact Hours

- **Lectures**: There are two sections. The lectures will be recorded and available on zoom. In person lecture attendance is optional.
- **Discussions**: There are six discussion sections. You are free to attend whichever you would like.
- **Office Hours**: Office hours are your time to ask questions one-on-one with course staff and get clarification on concepts from the course. We encourage you to go to GSI office hours for implementation questions about the homework and faculty office hours for conceptual questions.
- **Piazza**: The primary way to communicate with the course staff is through Piazza. The link is on canvas. We will use Piazza to make announcements about the course, such as homework releases or course project details. If you have questions about course concepts, need help with the homework, or have questions about course logistics, please post on Piazza instead of emailing course staff directly. Since Piazza is a shared discussion forum, asking and answering questions there is encouraged. On the other hand, please **do not post homework solutions on Piazza**. If you have questions about a particular piece of code, please make a private post.
- **Email**: If you need to discuss a sensitive matter that you would prefer not to be shared with the entire course staff, then please email the instructor or your section's GSI/IA directly.

## Course Policies

### Formatting and Submission
Submissions that do not follow these rules (and any additional ones specified in the homeworks) will get a 0.
- **No handwriting** - LaTeX is not required, but encouraged. Just put some effort into generating a readable PDF.
- **Mark answers on Gradescope** - With a few hundred students, graders will not have time to search for answers.

### Collaboration and External Sources
- **Automated plagiarism detection**: The vast majority of students are honorable. To ensure that honorable behavior is the incentivized behavior, we will run MOSS on the submitted homework.
- **Collaboration with students**: You should never know the specific implementation details of anyone else's homework or see their code. Working in teams and giving general advice about outputs or strategies (e.g., ‘‘if the image is really dark when you merge them together, you probably have screwed up the image mask with the number of images’’) is great. However, pair-programming or sitting next to someone else and debugging their code is not allowed.
- **Consulting outside material**: You can and should turn to other documentation (suggested textbooks, other professors’ lecture notes or slides, documentation from libraries). You may not read a set of code (pseudocode is fine). If you come across code in your search, close the window and don't worry about it.
- **Things you should never worry about**: Reading the documentation for publicly available libraries; clarifying ambiguities and mistakes in assignments, slides, handouts, textbooks, or documentation; discussion the general material; helping with things like cryptic numpy errors that are not related to class but part of the cost of doing business with a library; discussing the assignments to better understand what's expected and general solution strategies; discussing the starter code; discussing general strategies for writing and debugging code.
- **Generative AI**: Tools like ChatGPT are strongly discouraged. We know we can't stop you, however, using them will lead to you getting very little hands-on coding ability from this course and you will struggle on the midterm. The libraries used in this course are industry-standard and it is very helpful to be comfortable with them.

### Late Submissions
Our policy is quite generous. Exceptions will be made in only truly exceptional circumstances by the professor.
- **Late Days** - **6 total** late days across all homeworks. These will be applied automatically, no need to contact us. Homeworks are due by 11:59:59 on the due date. Thus, the late day would start at 12:00:00.
- **Penalty** - If you have 0 late days available, any subsequent late submissions will receive a 10% max score reduction per day. For example, if you submit 3 days late, you can receive at most 70% credit.
- **Late Deadline** - Late submissions will be accepted until a week after the deadline.
- **Project** - No late submissions. Late days and penalties will not be applied. This will be due as late as we can take them while still delivering grades on time.

### Regrades
- **Method** - Please submit regrade requests through Gradescope.
- **Deadline** - Submit regrade requests **within 1 week** of grades being released.
- **Minor Regrades** - Regrade requests that concern minor judgement calls that change the grade by <= 1 point for a problem or by <= 3 points for the whole homework will not be considered. If you believe this may affect your grade at the end of the semester, contact the professor.
