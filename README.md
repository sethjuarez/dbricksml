# Machine Learning Workshop
The goal of this workshop is to get you acquainted with
Machine Learning and several of the tools, frameworks, and
processes one must follow to succeed.

# The Challenge
Our primitive digits consist of *mostly* horizontal and 
vertical lines.
![](synthetic.png?raw=true)
Our goal is to have a machine learning algorithm predict,
given a new *mostly* horizontal or vertical line, whether 
it **__is__** *mostly* horizontal or vertical.

# The Data
There are two methods of approaching getting the data:
1. Hercules Mode - Use the [raw images](linear.zip)
2. Not-Hercules Mode - Use a [csv](linear.csv)
3. I'm Lazy Mode - Use my [code](generate.py) to generate it...

# Basic Linear Model
We will start with a basic linear model and try to make it
better (it stinks).
1. Hercules Mode - Start from scratch
2. Not Hercules Mode - A [starter](starter.py)
3. Lazy..... (Don't do it, unles you don't have time) - A [notebook](https://notebooks.azure.com/sethjuarez/libraries/workshop/html/linear.ipynb?WT.mc_id=aiml-0000-sejuare)

# Make it better!
It didn't perform very well with a linear model even though the cost
function was optimized. In this step we will make it a true neural
network. You can either:
1. Fix the code you've already created
2. Use [Keras](https://keras.io/#getting-started-30-seconds-to-keras)
