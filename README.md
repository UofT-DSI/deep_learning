# Deep Learning

## Contents
* [Description](#description)
* [Learning Outcomes](#learning-outcomes)
* [Logistics](#logistics)
    + [Module Contacts](#module-contacts)
    + [Module Materials](#module-materials)
* [Delivery of Module](#delivery-of-module)
    + [How the Technical Facilitator will deliver](#how-the-technical-facilitator-will-deliver)
    + [Expectations](#expectations)
    + [Requirements](#requirements)
    + [Classes](#classes)
    + [Tutorial](#tutorial)
* [Class Schedule](#class-schedule)
    + [Class Topics](#class-topics)
* [Grading Scheme](#grading-scheme)
    - [Submitting Notebooks](#submitting-notebooks)
* [Resources](#resources)
    + [Documents](#documents)
    + [Videos](#videos)
    + [How to get help](#how-to-get-help)
* [Folder Structure](#folder-structure)
* [Acknowledgements](#acknowledgements)

## Description
This Deep Learning module offers both fundamental understanding and practical skills necessary to develop, implement, test, and validate various deep learning models. The curriculum delves into the core concepts of Deep Learning, emphasizing its application across diverse domains. Students will explore the intricacies of neural networks, backpropagation, and the advanced architectures used in image processing, natural language processing, and more.

## Learning Outcomes
By the end of this Deep Learning module, students will:
1. Apply principles of neural networks, including architectures like CNNs and RNNs.
2. Implement deep learning models for tasks in image processing, NLP, and recommendation systems.
3. Utilize advanced techniques such as sequence-to-sequence models and attention mechanisms.
4. Evaluate and address challenges in model training, imbalanced classification, and metric learning.
5. Use Keras and TensorFlow to emphasize reproducible research.
6. Explain the ethical implications of deep learning models effectively to diverse audiences.

## Logistics

### Module Contacts
**Questions can be submitted to the #questions channel on Slack**

* Technical Facilitator: **{Name}** {Pronouns}. Emails to the Technical Facilitator can be sent to {first_name.last_name}@mail.utoronto.ca.
* Learning Support Staff: **{Name}** {Pronouns}. Emails to the Technical Facilitator can be sent to {first_name.last_name}@mail.utoronto.ca.

### Module Materials
This module's materials are adapted from the Deep Learning module taught at [Master Year 2 Data Science IP-Paris](https://www.ip-paris.fr/education/masters/mention-mathematiques-appliquees-statistiques/master-year-2-data-science). The module includes comprehensive lectures and lab notebooks covering fundamental and advanced topics in Deep Learning. While there is no designated textbook for this module, the adapted materials provide a thorough exploration of the subject, incorporating a blend of theoretical knowledge and practical applications.

## Delivery of Module
The module will run synchronously three times a week on Zoom. The first three days are used as "lectures" and will last a maximum of 3 hours. During this time, the Technical Facilitator will introduce the concepts for the week. The last day is used as an optional, asynchronous work period. The work periods will also last for up to 3 hours. During the last day, a Technical Facilitator or TA will be present on Zoom to assist learners reach the intended learning outcomes.

### How the Technical Facilitator will deliver
The Technical Facilitators will introduce the concepts through a collaborative live coding session using the Python notebooks found under `/01_slides`. All Technical Facilitators will also upload any live coding files to this repository for any learners to revisit under `/live_code`.
 
### Expectations
Learners are encouraged to be active participants while coding and are encouraged to ask questions throughout the module.
 
### Requirements
* Learners are not expected to have any coding experience, we designed the learning content for beginners.
* Learners are encouraged to ask questions and collaborate with others to enhance learning.
* Learners must have a computer and an internet connection to participate in online activities.
* Learners must have VSCode installed with the following extensions: 
    * [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
    * [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
* Learners must not use generative AI such as ChatGPT to generate code to complete assignments. It should be used as a supportive tool to seek out answers to questions you may have.
* We expect learners to have completed the [onboarding repo](https://github.com/UofT-DSI/onboarding/tree/main/onboarding_documents).
* Webcam usage is optional although highly encouraged. We understand that not everyone may have the space at home to have the camera on.

### Tutorial
Tutorial sessions are on the same date as each class. Tutorials will take place 30 minutes before and after each session. Tutorial attendance is optional, and organization is unstructured. The tutorial is the best place for questions/issues pertaining to software, labs, and assignments.

## Class Schedule

The module spans two weeks with a total of 6 classes, each from 6 PM to 8:30 PM EST. Below is the class schedule formatted as a table:


Classes will include lectures using prepared slides and live coding sessions. All slides will be accessible online prior to lectures. Students should actively participate in coding alongside the Technical Facilitator in real time and are encouraged to ask questions. 

### Class Topics

| Class | Date                | Topic                                                     | Slides                                     | Workbooks                                                                                                           | Suggested Additional Material                                                                                          |
|-------|---------------------|-----------------------------------------------------------|--------------------------------------------|---------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| 1     | TBD    | Introduction to Deep Learning                             | [Lecture 1 Slides](slides/Lecture_1.pdf)   | [Lab 1 Workbook](https://github.com/UofT-DSI/deep_learning/blob/main/02_labs/lab_1.ipynb)   | |
| 2     | TBD   | Neural Networks and Backpropagation                       | [Lecture 2 Slides](slides/Lecture_2.pdf)   | [Lab 2 Workbook](https://github.com/UofT-DSI/deep_learning/blob/main/02_labs/lab_2.ipynb)   | [3Blue1Brown Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)|
| 3     | TBD | Embeddings and Recommender Systems                        | [Lecture 3 Slides](slides/Lecture_3.pdf)   | [Lab 3 Workbook](https://github.com/UofT-DSI/deep_learning/blob/main/02_labs/lab_3.ipynb)   | |
| 4     | TBD   | Convolutional Neural Networks for Image Classification    | [Lecture 4 Slides](slides/Lecture_4.pdf)   | [Lab 4 Workbook](https://github.com/UofT-DSI/deep_learning/blob/main/02_labs/lab_4.ipynb)   | |
| 5     | TBD     | Deep Learning for Object Detection and Image Segmentation | [Lecture 5 Slides](slides/Lecture_5.pdf)   | [Lab 5 Workbook](https://github.com/UofT-DSI/deep_learning/blob/main/02_labs/lab_5.ipynb)   | |
| 6     | TBD    | Recurrent Neural Networks and NLP                         | [Lecture 6 Slides](slides/Lecture_6.pdf)   | [Lab 6 Workbook](https://github.com/UofT-DSI/deep_learning/blob/main/02_labs/lab_6.ipynb)   ||


## Grading Scheme

The grading for this module is based on two components: assignments and class participation, including the completion of Jupyter notebooks. The grading scheme is as follows:

| Assessment       | Number | Individual Weight | Cumulative Weight |
|------------------|--------|-------------------|-------------------|
| Assignments      | 2      | 35%               | 70%               |
| Jupyter Notebooks | 10     | 2%                | 20%               |
| Participation    | NA     | NA                | 10%               |

- Assignments consist of two major tasks completed at the end of the first two weeks.
- Jupyter Notebooks are to be completed throughout the module. Completion of these notebooks is pass/fail.
- Participation includes engagement in class discussions, activities, and overall contribution to the module environment.


**Assignments**

Assignments are a vital part of this module, focusing on the application of deep learning concepts. Two main assignments are scheduled, one at the end of each of the first two weeks. These assignments will be introduced in class and can be discussed with the Technical Facilitator or TA during office hours or via email. They should be completed independently and submitted through the designated Google Forms links, following the naming convention `firstname_lastname_a#`. Please request extensions well in advance.

| Assessment   | Link                                                                                                                      | Due Date                                     | Submission Link                                     |
|--------------|---------------------------------------------------------------------------------------------------------------------------|----------------------------------------------|-----------------------------------------------------|
| Assignment 1 | [Open in Colab](https://github.com/UofT-DSI/deep_learning/blob/main/03_assignments/assignment_1.ipynb) | TBD     | Submission Closed                                   |
| Assignment 2 | [Open in Colab](https://github.com/UofT-DSI/deep_learning/blob/main/03_assignments/assignment_2.ipynb) | TBD | [Submit Here](https://forms.gle/5Z6AsKxx6vURxZkK8)  |
| Workbooks    |                             | TBD | [Submit Here](https://forms.gle/T6wTeZRQ4ZnEXRGGA)  |                                              |                                                    |

You may submit assignments multiple times before the deadline. The last submission will be graded.

**Notebook Completion**

Students are expected to complete the Jupyter notebooks associated with each class. Completion includes actively coding along with the Technical Facilitator and answering any questions in the notebooks. These notebooks are due by the end of the module, but it is highly recommended to complete them as you progress through the material to stay on top of the content. Notebooks are to be submitted for pass/fail grading.

#### Submitting Notebooks

Notebooks are to be submitted together at the end of the module. To submit, please follow these steps:

1. Create a folder named `firstname_lastname_notebooks` and place all completed notebooks inside.
2. Compress the folder into a `.zip` file.
3. Upload the `.zip` file to [this form](https://forms.gle/T6wTeZRQ4ZnEXRGGA).

You may submit notebooks multiple times before the deadline. The last submission will be graded.

**Note:** If any content in the assignments or notebooks is related to topics not covered in class due to schedule changes, those parts will be excluded from grading. Such exclusions will be clearly communicated before the assignment due date.

**Participation**

We hope all members of the module regularly participate. We define participation broadly and include attendance, asking questions, answering others' questions, participating in discussions, etc.

## Resources
Feel free to use the following as resources:

### Documents
- [Cheatsheet](https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-deep-learning)
- [Keras Cheatsheet](https://www.datacamp.com/cheat-sheet/keras-cheat-sheet-neural-networks-in-python)

### Videos
- [What is Deep Learning?](https://www.youtube.com/watch?v=6M5VXKLf4D4)
- [Neural Network in 5 minutes](https://www.youtube.com/watch?v=bfmFfD2RIcg)
- [What is NLP?](https://www.youtube.com/watch?v=CMrHM8a3hqw)
- [Classification and Regression in Machine Learning](https://www.youtube.com/watch?v=TJveOYsK6MY)
- [Supervised vs Unsupervised vs Reinforcement Learning](https://www.youtube.com/watch?v=1FZ0A1QCMWc)
- [3Blue1Brown Neural Networks Playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

### How to get help
![image](./steps_to_ask_for_help.png)

## Folder Structure

```markdown
.
├── 01_slides
├── 02_labs
├── 03_assignments
├── 04_instructional_team
├── 05_additional_resources
├── LICENSE
├── README.md
└── steps_to_ask_for_help.png
```

* **slides:** module slides as PDF files
* **labs:** Interactive notebooks to be done after each lecture (.ipynb files)
* **live_coding:** Notebooks from class live coding sessions
* **assignments:** Graded assignments
* **data**: Contains all data associated with the module
* **instructors:** Instructions for the Technical Facilitator on what to teach
* README: This file!
* .gitignore: Files to exclude from this folder, specified by the Technical Facilitator

## Acknowledgement

We wish to acknowledge this land on which the University of Toronto operates. For thousands of years, it has been the traditional land of the Huron-Wendat, the Seneca, and most recently, the Mississaugas of the Credit River. Today, this meeting place is still the home to many Indigenous people from across Turtle Island and we are grateful to have the opportunity to work on this land.