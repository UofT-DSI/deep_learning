# Deep Learning

## Contents
* [Description](#description)
* [Learning Outcomes](#learning-outcomes)
* [Assignments](#assignments)
* [Contacts](#contacts)
* [Delivery of the Learning Module](#delivery-of-the-learning-module)
* [Schedule](#schedule)
* [Requirements](#requirements)
* [Resources](#resources)
    + [Documents](#documents)
    + [Videos](#videos)
    + [How to get help](#how-to-get-help)
* [Folder Structure](#folder-structure)

## Description
This module offers both fundamental understanding and practical skills necessary to develop, implement, test, and validate various deep learning models. The curriculum delves into the core concepts of Deep Learning, emphasizing its application across diverse domains. Participants will explore the intricacies of neural networks, backpropagation, and the advanced architectures used in image processing, natural language processing, and more.

## Learning Outcomes
By the end of this learning module, participants will be able to:
1. Apply principles of neural networks, including architectures like CNNs and RNNs.
2. Implement deep learning models for tasks in image processing, NLP, and recommendation systems.
3. Utilize advanced techniques such as sequence-to-sequence models and attention mechanisms.
4. Evaluate and address challenges in model training, imbalanced classification, and metric learning.
5. Use Keras and TensorFlow to emphasize reproducible research.
6. Explain the ethical implications of deep learning models effectively to diverse audiences.

## Assignments

The assessment for this module is based on two components: assignments and class participation, including the completion of Jupyter notebooks.

| Assessment        | Number | Individual Weight | Cumulative Weight |
| ----------------- | ------ | ----------------- | ----------------- |
| Assignments       | 2      | 35%               | 70%               |
| Jupyter Notebooks | 10     | 2%                | 20%               |
| Participation     |        |                   | 10%               |

- Assignments consist of two major tasks completed at the end of the first two weeks.
- Jupyter Notebooks are to be completed throughout the module. Completion of these notebooks is pass/fail.
- Participation includes engagement in class discussions, activities, and overall contribution to the module environment.

**Assignments**

Assignments are a vital part of this module, focusing on the application of deep learning concepts. Two main assignments are scheduled, one at the end of each of the two weeks. These assignments will be introduced in live session and can be discussed with the Technical Facilitator or Learning Support during office hours, work periods or via Slack. They should be completed independently.

| Assessment   | Link                                            | Due Date |
| ------------ | ----------------------------------------------- | -------- |
| Assignment 1 | [Notebook](./02_activities/assignments/assignment_1.ipynb) | Sunday, July 6th      |
| Week 1 Workbooks    |                                                 | Sunday, July 6th      |
| Assignment 2 | [Notebook](./02_activities/assignments/assignment_2.ipynb) | Friday, July 18th      |
| Week 2+3 Workbooks    |                                                 | Friday, July 18th      |


You may submit assignments multiple times before the deadline. The last submission will be graded.

**Notebook Completion**

Participants are expected to complete the Jupyter notebooks associated with each session. Completion includes actively coding along with the Technical Facilitator and answering any questions in the notebooks. These notebooks are due by the end of each week, but it is highly recommended to complete them as you progress through the material to stay on top of the content. Notebooks are to be submitted for pass/fail assessment.

**Submitting Notebooks**

Notebooks are to be submitted together at the end of each week.
You may submit notebooks multiple times before the deadline. The last submission will be graded.

**Participation**

We hope all members of the module regularly participate. We define participation broadly and include attendance, asking questions, answering others' questions, participating in discussions, etc.

## Contacts
**Questions can be submitted to the _#cohort-6-ml-help_ channel on Slack**

* Technical Facilitator: **Alex Olson** he/him. Emails to the Technical Facilitator can be sent to alex.olson@utoronto.ca.
* Learning Support Staff: 
  * **Emma Teng** e.teng@mail.utoronto.ca
  * **Tianyi Liu** tianyi@psi.toronto.edu
  * **Edward Chen** edwardty.chen@utoronto.ca
  

## Delivery of the Learning Module
This module will include live learning sessions and optional, asynchronous work periods. During live learning sessions, the Technical Facilitator will introduce and explain key concepts and demonstrate core skills. Learning is facilitated during this time. Before and after each live learning session, the instructional team will be available for questions related to the core concepts of the module. The Technical Facilitator will introduce concepts through a collaborative live coding session using the Python notebooks found under `/01_materials/slides`. The Technical Facilitator will also upload live coding files to this repository for participants to revisit under `./04_cohort_six/live_code`.

Optional work periods are to be used to seek help from peers, the Learning Support team, and to work through the homework and assignments in the learning module, with access to live help. Content is not facilitated, but rather this time should be driven by participants. We encourage participants to come to these work periods with questions and problems to work through. 
 
Participants are encouraged to engage actively during the learning module. They key to developing the core skills in each learning module is through practice. The more participants engage in coding along with the instructional team, and applying the skills in each module, the more likely it is that these skills will solidify. 

This module's materials are adapted from the Deep Learning module taught at [Master Year 2 Data Science IP-Paris](https://www.ip-paris.fr/education/masters/mention-mathematiques-appliquees-statistiques/master-year-2-data-science). The module includes comprehensive lectures and lab notebooks covering fundamental and advanced topics in Deep Learning. While there is no designated textbook for this module, the adapted materials provide a thorough exploration of the subject, incorporating a blend of theoretical knowledge and practical applications.

## Schedule

| Live Learning Session | Date | Topic                                                     | Slides                                                                                | Workbooks                               | Suggested Additional Material                                                                           |
| ----- | ---- | --------------------------------------------------------- | ------------------------------------------------------------------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| 1     | Wednesday, July 2nd  | Introduction to Deep Learning                             | [Slides](./01_materials/slides/01_introduction.pdf)                                   | [Lab 1 Workbook](./01_materials/labs/lab_1.ipynb) |                                                                                                         |
| 2     | Thursday, July 3rd  | Neural Networks and Backpropagation                       | [Slides](./01_materials/slides/02_neural_networks_and_backpropagation.pdf)            | [Lab 2 Workbook](./01_materials/labs/lab_2.ipynb) | [3Blue1Brown Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) |
| 3     | Tuesday, July 8th  | Embeddings and Recommender Systems                        | [Slides](./01_materials/slides/03_recommender_systems_&_embeddings.pdf)               | [Lab 3 Workbook](./01_materials/labs/lab_3.ipynb) |                                                                                                         |
| 4     | Wednesday, July 9th  | Convolutional Neural Networks for Image Classification    | [Slides](./01_materials/slides/04_convolutional_neural_networks.pdf)                  | [Lab 4 Workbook](./01_materials/labs/lab_4.ipynb) |                                                                                                         |
| 5     | Thursday, July 10th  | Deep Learning for Object Detection and Image Segmentation | [Slides](./01_materials/slides/05_convolutional_neural_networks_part_II.pdf)          | [Lab 5 Workbook](./01_materials/labs/lab_5.ipynb) |                                                                                                         |
| 6     | Tuesday, July 15th  | Recurrent Neural Networks and NLP                         | [Slides](./01_materials/slides/06_natural_language_processing_with_deep_learning.pdf) | [Lab 6 Workbook](./01_materials/labs/lab_6.ipynb) |                                                                                                         |
 
### Requirements
* Participants are expected to have completed Shell, Git, Python, Linear Regression, Classification, and Resampling, Production, and Algorithms & Data Structures learning modules.
* Participants are encouraged to ask questions, and collaborate with others to enhance their learning experience.
* Participants must have a computer and an internet connection to participate in online activities.
* Participants must not use generative AI such as ChatGPT to generate code in order to complete assignments. It should be used as a supportive tool to seek out answers to questions you may have.
* We expect participants to have completed the instructions mentioned in the [onboarding repo](https://github.com/UofT-DSI/onboarding/).
* We encourage participants to default to having their camera on at all times, and turning the camera off only as needed. This will greatly enhance the learning experience for all participants and provides real-time feedback for the instructional team. 
* Participants must have VSCode installed with the following extensions: 
    * [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
    * [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)

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

### How to Get Help
#### 1. Gather information about your problem
- Copy and paste your error message
- Copy and paste the code that caused the error, and the last few commands leading up to the error
- Write down what you are trying to accomplish with your code. Include both the specific action, and the bigger picture and context
- (optional) Take a screenshot of your entire workspace

#### 2. Try searching the web for your error message
- Sometimes, the error has common solutions that can be easy to find!
   - This will be faster than waiting for an answer
- If none of the solutions apply, consider asking a Generative AI tool
   - Paste your code, the error message, and a description of your overall goals

#### 3. Try asking in your cohort's Slack help channel
- Since we're all working through the same material, there's a good chance one of your peers has encountered the same error, or has already solved it
- Try searching in the DSI Certificates Slack help channel for whether a similar query has been posted
- If the question has not yet been answered, post your question!
   - Describe your the overall goals, the context, and the specific details of what you were trying to accomplish
   - Make sure to **copy and paste** your code, your error message
   - Copying and pasting helps:
      1. Your peers and teaching team quickly try out your code
      1. Others to find your question in the future

#### Great resources on how to ask good technical questions that get useful answers
- [Asking for Help - The Odin Project](https://www.theodinproject.com/lessons/foundations-asking-for-help)
- [How do I ask a good question? - Stack Overflow](https://stackoverflow.com/help/how-to-ask)
- [The XY problem: A question pitfall that won't get useful answers](https://xyproblem.info/)
- [How to create a minimal reproducible example](https://stackoverflow.com/help/minimal-reproducible-example)

## Folder Structure

```markdown
.
├── .github
├── 01_materials
├── 02_activities
├── 03_instructional_team
├── 04_cohort_six
├── .gitignore
├── LICENSE
└── README.md
```

* **.github**: Contains issue templates and pull request templates for the repository.
 * **materials**: Module slides and interactive notebooks (.ipynb files) used during learning sessions.
 * **activities**: Contains graded assignments, exercises, and homework to practice concepts covered in the learning module.
 * **instructional_team**: Resources for the instructional team.
 * **cohort_six**: Additional materials and resources for cohort six.
 * **.gitignore**: Files to exclude from this folder, specified by the Technical Facilitator
 * **LICENSE**: The license for this repository.
 * **README.md**: This file.
 
