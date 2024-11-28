# K-Means Clustering for Tweets

This project implements a K-Means clustering algorithm for processing tweets. It cleans and clusters tweets based on the Jaccard distance between tweet text.

## Prerequisites

Make sure you have Python 3.6 or higher installed on your system. You can download it from [here](https://www.python.org/downloads/).

### Dependencies

You will need the following Python libraries for the project:

- `chardet`: For detecting file encodings.
- `html`: For decoding HTML entities in tweets.
- `re`: For regular expression-based text cleaning.
- `random`: For random sampling of tweets.
- `collections`: For counting cluster sizes.

You can install the required libraries using `pip`. Create a virtual environment (recommended) and install the dependencies:

```bash
# Create a virtual environment (optional but recommended)
python3 -m venv venv
# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate
```
### Install the required packages
```
pip install -r requirements.txt
```

## Running the File

To run the main clustering script, follow these steps:
1. Clone the repository or download the files to your local machine.
2.	Make sure you have your dataset (e.g., **usnewshealth.txt**) in the same directory as the main.py file or specify the correct path in the code.
3.	Open a terminal or command prompt in the project directory.
4.	Run the main.py file with Python:
    ```
    python main.py
    ```