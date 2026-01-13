---
layout: spec
---

<link href="style.css" rel="stylesheet">

# Overview

With the impressive recent performance of machine-generated visual content, studying how to create realistic imagery
using traditional and AI-based tools is becoming increasingly important.
This course will introduce students to the theoretical and practical foundations of computer graphics, as well as the
recent advances in generative models to automate the content creation process.
This course is designed to prepare both undergraduate and graduate students to learn how visual content can be created
and to prepare for conducting research in a relevant area.

- **Instructor**: Jeong Joon Park
- **Student Assistants**:
  - Xingpeng Xia (Grad)
  - Yixing Wang (Grad)
  - Aidan Donley (Undergrad)
  - Julian Whittaker (Undergrad)
- Course email: **<um-graphics@umich.edu>** (Don't email the instructor or the individual students.)
- **Lecture**: Monday/Wednesday 10:30 AM - 12:00 Noon, 3336 DC
- **Labs**: Tuesday/Thursday 3:30 PM - 5:30 PM, 1500 EECS.
- [**Piazza Signup**](https://piazza.com/umich/winter2026/eecs498598016)
- [**GradeScope**](https://www.gradescope.com/courses/1212518)
- [**Lecture Recordings**](https://leccap.engin.umich.edu/leccap/site/t0hbic0mhfa1248530c) (It could take a couple of days to be processed and uploaded)
- Note on Waitlist/Overriding: We will accept all interested students. Send a note to the course email for override and talk to the CSE advising office to register.

# Homeworks

- [Homework 1: Rasterization](). Due Date: February 7th.
- [Homework 2: Ray Tracing](). Due Date: February 28th.
- [Homework 3: Blender](). Due Date: March 14th.
- [Homework 4: 3D Recon](). Due Date: March 30th.
- [Homework 5: Generative Model](). Due Date: April 22nd. No Late Day.

Since this is only the second iteration of this course, there can be errors in the homeworks, and will be fixed as soon as we spot them. If you encounter problems that you think are not caused by your implementation, first check whether your codebase is up to date with the published ones on github. You are welcome to come to the lab sessions for reporting or clarification of such issues.


# Schedule

|# | Date               | Topic                     | Material    |
|--|--------------------|---------------------------|-------------|
| 1| Wednesday, Jan. 7  | Introduction              | [Slides](https://drive.google.com/file/d/12Cq-AGQLzVFv1KqNi_8jZSxC0MQmZdIV/view?usp=drive_link) |
| 2| Monday, Jan. 12    | Transformation 1          | [Slides](https://drive.google.com/file/d/1EugbqVAtTRotGEg10HB35mzrczIX5X9G/view?usp=drive_link) |
| 3| Wednesday, Jan. 14 | Transformation 2          |             |
|  | Monday, Jan. 19    | Martin Luther King Jr. Day| Slides (TBD)|
| 4| Wednesday, Jan. 21 | Rasterization 1           |             |
| 5| Monday, Jan. 26    | Rasterization 2           |             |
| 6| Wednesday, Jan. 28 | Rasterization 3           |             |
| 7| Monday, Feb. 2     | Rasterization 4           |             |
| 8| Wednesday, Feb. 4  | Ray Tracing 1             |             |
| 9| Monday, Feb. 9     | Ray Tracing 2             |             |
|10| Wednesday, Feb. 11 | Ray Tracing 3             |             |
|11| Monday, Feb. 16    | Ray Tracing 4             |             |
|12| Wednesday, Feb. 18 | Advanced Topics           |             |
|13| Monday, Feb. 23    | Geometry                  |             |
|14| Wednesday, Feb. 25 | Reconstruction            |             |
|  | Monday, Mar. 2     | Spring Break              | Slides (TBD)|
|  | Wednesday, Mar. 4  | Spring Break              | Slides (TBD)|
|15| Monday, Mar. 9     | Representations           |             |
|16| Wednesday, Mar. 11 | Neural Fields 1           |             |
|17| Monday, Mar. 16    | Neural Fields 2           |             |
|18| Wednesday, Mar. 18 | Neural Fields 3           |             |
|19| Monday, Mar. 23    | Neural Fields 4           |             |
|20| Wednesday, Mar. 25 | Generative Models 1       |             |
|21| Monday, Mar. 30    | Generative Models 2       |             |
|22| Wednesday, Apr. 1  | Generative Models 3       |             |
|23| Monday, Apr. 6     | Generative Models 4       |             |
|24| Wednesday, Apr. 8  | Guest Lecture <br>(TBD)   |             |
|25| Monday, Apr. 13    | No Class                  | Slides (TBD)|
|26| Wednesday, Apr. 15 | Guest Lecture <br>(TBD)   |             |
|27| Monday, Apr. 20    | Final Exam                | Slides (TBD)|

# Syllabus

## Scope and Topics

This course is divided into three parts.
The first part discusses the fundamentals of graphics, including camera models, rasterization, materials and lighting,
ray-tracing, geometry modeling, and texture modeling.
The second and third part discuss automating the traditional graphics pipeline using artificial intelligence.
The second part focuses on 3D reconstruction, including structure-from-motion, neural radiance fields (NeRF), Gaussian
Splatting, etc. The third part discusses theories and practices of generative models, including generative adversarial
networks (GANs) and diffusion models and their recent applications to 3D content creation. Each of these topics would
involve homework assignments involving individual programming.

## Prerequisites

Students taking this course should be comfortable with mathematical expositions and proofs.
Familiarity with linear algebra, probability, and multivariate calculus is required.
Formally, the students are expected to take (*EECS 281*) and (*MATH 425* or *412* or *EECS 301* or *IOE 265* or *TO 301*)
and (*EECS 351* or *MATH 214* or *MATH 217* or *296* or *417* or *419* or *ROB 101*), or be in graduate standings before registering for this course.
The generative parts of the course will involve significant deep learning and computer vision, including neural network design
and training.
Thus, familiarity with deep learning, especially in the context of computer vision, is highly recommended.

## Difference between 498/598

While all undergraduate and graduate students will join the same lectures, they will be given the same
homework assignments or exams. The final grades will be assigned within the two tracks.

## Class Participation

This course expects a great amount of class participation from the students and a significant part of the final grade will be
from the participation.
Specifically, **students are expected to join the lectures in person and participate in in-class discussions and quizzes that happen after each lecture**.
Students will be expected to contribute to online Q&As and discussions.
We will track student participation in the lab sessions and the Piazza discussions and reflect them to the grading accordingly.

- There will be **in-class quizzes** at the end of each course, whose results are reflected in the participation grade. The students are expected to have discussions with peers and upload their answers to Gradescope.

## Lab sessions

T/TH lab sessions will be a more involved version of office hours and interchangeable between the two time slots.
Some of the TAs and instructors will be present to help students with the conceptual and homework-related questions,
including hands-on programming aids.
We hope it will also serve as a bazaar of knowledge where students exchange ideas/information and help each other.

## Piazza

- We will use Piazza for discussions on conceptual and technical questions **among classmates**. TAs will answer technical questions during the offline lab sessions.
- We expect the students to actively participate in Piazza discussions, which will be a significant part of the participation score.
- Please check Piazza for already posted questions before posting a new one. Unnecessarily clogging up Piazza makes the platform less usable for everybody.
- Please use Piazza for all communications as much as possible.  Others will benefit from answers and discussions on public questions.

## Grading

Your grade will be based on:

- **Programming Assignments (60%)**: There will be five (5) individual programming assignments.
  - Each assignment is worth 12%.
  - Including a Blender project.
- **Final Exam (25%)**.
- **Participation (15%)**.
  - End-of-class discussion, lab session, and Piazza participation.
- **Total (100%)**.

## Course Policies

### Formatting and Submission

Submissions that do not follow these rules (and any additional ones specified in the homeworks) will get a 0.

- **No handwriting** - LaTeX is not required, but encouraged. Just put some effort into generating a readable PDF.
- **Make sure your code compiles**. This applies especially to C++ (the first two) homeworks. Grading will be conducted based on the output of your program.

### Collaboration and External Sources

- **Automated plagiarism detection**: The vast majority of students are honorable. To ensure that honorable behavior is the incentivized behavior, we will run MOSS on the submitted homework.
- **Collaboration with students**: We strongly encourage you to collaborate with other students. We're quite generous about the collaboration policy: you're free to collaborate as much as you want with other students, including discussions and pair programming. However, after your collaboration, you should write your code **yourself**. This means that you should discard all of the code you wrote with your peers and write them **again** yourself. We will check code similarity between submissions, and notify students with too similar implementations (MOSS).
- **Consulting outside material**: You can and should turn to other documentation (suggested textbooks, other professors’ lecture notes or slides, documentation from libraries). You may not read a set of code (pseudocode is fine). If you come across code in your search, close the window and don't worry about it.
- **Generative AI**: Tools like ChatGPT are strongly discouraged. You can use them for general concepts, but you are prohibited from using them for code generation.

### Late Submissions

Our policy is generous. Late homework will be deducted 10% flat rate for 10 days. After that, we will impose 25% penalty (submission more than 10 days late).

### Regrades

- **Method** - Please submit regrade requests through Gradescope.
- **Deadline** - Submit regrade requests **within 1 week** of grades being released.
- **Minor Regrades** - Regrade requests that concern minor judgement calls that change the grade by <= 1 point for a problem or by <= 3 points for the whole homework will not be considered. If you believe this may affect your grade at the end of the semester, contact the course email.

## Exams

There will be a final exam.

## Textbooks

There is no required textbook. However, the following textbook is recommended for the course:

- *Fundamentals of Computer Graphics 4th Edition. Marschner & Shirley*

## Topics Covered

<ol>
  <li>Classical Computer Graphics – 12 sessions</li>
  <ol style="list-style-type: lower-alpha !important;">
    <li>Camera models. Homogenous coordinates. Transformations</li>
    <li>Rasterization</li>
    <li>Materials and lighting</li>
    <li>Ray-tracing. Global Illuminations</li>
    <li>Texture mapping</li>
    <li>Blender project</li>
  </ol>
  <li>3D Reconstruction (for content capture) – 4 sessions</li>
  <ol style="list-style-type: lower-alpha !important;">
    <li>Structure from motions</li>
    <li>Neural implicit representations (neural fields)</li>
    <li>Neural radiance fields (NeRF)</li>
    <li>NeRF extensions (hybrid representations, dynamic reconstructions, etc)</li>
  </ol>
  <li>Generative Models (for content creation) – 6 sessions</li>
  <ol style="list-style-type: lower-alpha !important;">
    <li>Variational autoencoders</li>
    <li>Generative adversarial networks (GANs)</li>
    <li>Diffusion and score-based models</li>
    <li>Flow-based models</li>
    <li>Inverse problem-solving with diffusion models</li>
  </ol>
  <li>3D Generative Models (for content creation) – 4 sessions</li>
  <ol style="list-style-type: lower-alpha !important;">
    <li>3D GANs</li>
    <li>Distilling diffusion models for 3D generation</li>
    <li>Other 3D generative models</li>
  </ol>
</ol>
