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
  - Ang Cao (Grad)
  - Boyang Wang (Grad)
  - Junwen Yu (Undergrad)
  - Yuqi Meng (Undergrad)
- Course email: **um-graphics@umich.edu** (Don't email the instructor or the individual students.)
- **Lecture**: Monday/Wednesday 10:30 AM - 12:00 Noon, 1109 FXB
- **Lab Hours**:
	- Tuesday/Thursday 3:30 PM - 5:00 PM, 138 NAME/107 GFL, respectively.
- [**Piazza Signup**](https://piazza.com/umich/fall2024/eecs498598014)
- [**GradeScope**](https://www.gradescope.com/courses/833954)
- [**Lecture Recordings**](https://leccap.engin.umich.edu/leccap/site/cqusadkfhu4csgkbyjm) (It could take a couple of days to be processed and uploaded)
- Note on Waitlist/Overriding: We will accept all interested students. Send a note to the course email for override and talk to the CSE advising office to register.

# Homeworks
- [Homework 1: Rasterization](https://drive.google.com/file/d/1SjF8xr_MgllB17Oh4Rj0uWw1uaKaJQpp/view). Due Date: September 20th. 
- [Homework 2: Ray Tracing](https://drive.google.com/file/d/1d4nHJt09riBWm015bod86KB87xnJ2obb/view?usp=sharing). Due Date: October 11th.

Since this is the first iteration of this course, there can be errors in the homeworks, and will be fixed as soon as we spot them. If you encounter problems that you think are not caused by your implementation, first check whether your codebase is up to date with the published ones on github. You are welcome to come to the lab sessions for reporting or clarification of such issues.

# Schedule

|# | Date               | Topic      | Material    |
|--|--------------------|------------|-------------|
| 1| Monday, Aug. 26    | Introduction| [Slides](https://drive.google.com/file/d/16kWVZEFDluwPRDZLwnOwzFF7hh-1lC-M/view?usp=sharing)|
| 2| Wednesday, Aug. 28 | Transformation 1| [Slides](https://drive.google.com/file/d/1P3RphNs3s5MvwdrJbU3zc5grL3_wpNvh/view?usp=sharing)|
|  | Monday, Sep. 2     | Labor Day  | |
| 3| Wednesday, Sep. 4  | Transformation 2|[Slides](https://drive.google.com/file/d/1MrPkXfkmkEZmjPTURwmWFVUMEz_efk-P/view?usp=sharing)|
| 4| Monday, Sep. 9     | Rasterization 1| [Slides](https://drive.google.com/file/d/1SgzICWdcPcsiQpMl5bUKVn9u7mKUgcul/view?usp=sharing)|
| 5| Wednesday, Sep. 11 | Rasterization 2| [Slides](https://drive.google.com/file/d/1mIczCahnhmzTC0FBOA8RW0y4bz3f7CJ8/view?usp=sharing)|
| 6| Monday, Sep. 16    | Rasterization 3| [Slides](https://drive.google.com/file/d/1AXfPuvMipnjTptTlK6FnvElgBJt6QoBj/view?usp=sharing)|
| 7| Wednesday, Sep. 18 | Rasterization 4| [Slides](https://drive.google.com/file/d/170LrXqa03A0-JAZCvj5wp_ATEKG2a4YV/view?usp=sharing)|
| 8| Monday, Sep. 23    | Ray Tracing 1 |[Slides](https://drive.google.com/file/d/1SIn2o7d9HEvrT0CbC0m-m7zpvzldjKpT/view?usp=sharing)|
| 9| Wednesday, Sep. 25 | Ray Tracing 2| [Slides](https://drive.google.com/file/d/1HvNf7fWVpzqtkuHh48aoOKBGoFvQcFTs/view?usp=sharing) |
|10| Monday, Sep. 30    | Ray Tracing 3| Slides (TBD)|
|11| Wednesday, Oct. 2  | Ray Tracing 4| Slides (TBD)|
|12| Monday, Oct. 7     | Advanced Topics| Slides (TBD)|
|13| Wednesday, Oct. 9  | Geometry| Slides (TBD)|
|  | Monday, Oct. 14    | Fall Study Break | Slides (TBD)|
|14| Wednesday, Oct. 16 | No Class| Slides (TBD)|
|15| Monday, Oct. 21    | Reconstruction (TBD)| Slides (TBD)|
|16| Wednesday, Oct. 23 | Representations (TBD)| Slides (TBD)|
|17| Monday, Oct. 28    | Neural Fields 1 (TBD)| Slides (TBD)|
|18| Wednesday, Oct. 30 | Neural Fields 2 (TBD)| Slides (TBD)|
|19| Monday, Nov. 4     | Neural Fields 3 (TBD)| Slides (TBD)|
|20| Wednesday, Nov. 6  | Neural Fields 4 (TBD)| Slides (TBD)|
|21| Monday, Nov. 11    | Generative Models 1 (TBD)| Slides (TBD)|
|22| Wednesday, Nov. 13 | Generative Models 2 (TBD)| Slides (TBD)|
|23| Monday, Nov. 18    | Generative Models 3 (TBD)| Slides (TBD)|
|24| Wednesday, Nov. 20 | Generative Models 4 (TBD)| Slides (TBD)|
|25| Monday, Nov. 25    | Guest Lecture (TBD)| Slides (TBD)|
|  | Wednesday, Nov. 27 | Thanksgiving Break | Slides (TBD)|
|26| Monday, Dec. 2     | Final Exam| Slides (TBD)|
|27| Wednesday, Dec. 4  | Guest Lecture| Slides (TBD)|

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
There will be compensation for the students who participate actively in the final exam. See the grading scheme section for details.
- There will be **in-class quizzes** at the end of each course, whose results are reflected in the participation grade. The students are expected to have discussions with peers and upload their answers to Gradescope.
- Extra points will be given to the students who actively participate during Piazza discussions and in-person lab sessions. These extra points will be added to the final exam score.

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
- **Programming Assignments (60%)**: There will be five individual programming assignments.
  - **Including a Blender Project (12.5%)**.
- **Final Exam (25%)**.
- **Participation (15%+10%)**.
  - **10% extra credit added to the Final Exam score**
  - **Extra credit given to students with satisfactory participation**
- **Total (100%)**.

Note that we will allow the participation grade to overflow to the final exam. This means that if you get a perfect participation grade (15%+10%) the 10% score will be added to your final exam.

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
Our policy is generous. Late homework will be deducted 10% flat rate for 10 days. After that, we will impose 50% penalty (submission more than 10 days late).

### Regrades
- **Method** - Please submit regrade requests through Gradescope.
- **Deadline** - Submit regrade requests **within 1 week** of grades being released.
- **Minor Regrades** - Regrade requests that concern minor judgement calls that change the grade by <= 1 point for a problem or by <= 3 points for the whole homework will not be considered. If you believe this may affect your grade at the end of the semester, contact the course email.

## Exams
There will be a final exam.

## Textbooks

There is no required textbook. However, the following textbook is recommended for the course:

*Fundamentals of Computer Graphics 4th Edition. Marschner & Shirley*

## Topics Covered
- 1. Classical Computer Graphics – 12 sessions
  - a. Camera models. Homogenous coordinates. Transformations.
  - b. Rasterization.
  - c. Materials and lighting.
  - d. Ray-tracing. Global Illuminations.
  - e. Texture mapping.
  - f. Blender project.
- 2. 3D Reconstruction (for content capture) – 4 sessions
  - a. Structure from motions.
  - b. Neural implicit representations (neural fields)
  - c. Neural radiance fields (NeRF)
  - d. NeRF extensions (hybrid representations, dynamic reconstructions, etc)
- 3. Generative Models (for content creation) – 6 sessions
  - a. Variational autoencoders
  - b. Generative adversarial networks (GANs)
  - c. Diffusion and score-based models
  - d. Flow-based models
  - e. Inverse problem-solving with diffusion models
- 4. 3D Generative Models (for content creation) – 4 sessions
  - a. 3D GANs
  - b. Distilling diffusion models for 3D generation
  - c. Other 3D generative models
