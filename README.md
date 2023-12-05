# GLM Model Execution

This repository contains Python scripts and functions that demonstrate the implementation of a Generalized Linear Model (GLM). The GLM is used for predictive modeling tasks based on provided datasets. The code performs various data preprocessing steps, feature engineering, model training, and evaluation using different Python libraries.

## Overview

The repository includes the following files:

- `GLM_Model_26.py`: Python script containing functions for GLM execution, data preprocessing, and model evaluation.
- `send_request.py`: Script to send requests to the Flask app for running the GLM model.
- `app.py`: Flask application to execute the GLM model via HTTP requests.
- Other supporting files and directories: Additional scripts, data files, or resources used by the main scripts.

## Requirements

To run the code, ensure you have:

- Python installed (recommended version: 3.x).
- Required Python libraries (`numpy`, `pandas`, `scikit-learn`, `statsmodels`, `bokeh`, `matplotlib`, `seaborn`, etc.).

## Usage

1. Clone or download the repository to your local machine.

2. Open a terminal or command prompt and navigate to the directory containing the downloaded files.
#### `pip install -r requirements.txt`

3. Run the `GLM_Model_26.py` script with appropriate arguments specifying the paths to the train and test data files.
#### `python GLM_Model_26.py exercise_26_train.csv exercise_26_test.csv`

4. To interact with the Flask app and execute the model, run the `send_request.py` script. Ensure the Flask app (`app.py`) is running. Example:
python send_request.py (This is still under development)


## Additional Notes

- Provide the correct file paths for the train and test data files as command-line arguments.
- Review the comments within the code for function descriptions and specific details about each step.
- For any issues or inquiries, please contact [arjunsarihyan@gmail.com].
