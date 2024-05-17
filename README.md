 Student Performance Predictor

A Streamlit application for predicting student performance based on various features such as gender, ethnicity, parental education level, lunch type, test preparation course, and scores in math, reading, and writing.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

This Streamlit app allows users to input student data and receive predictions on the student's performance group (below-average, average, high achievers) along with recommendations for improvement and visualization of the predicted result. The app uses a pre-trained KMeans clustering model.

## Features

- Input student data through a user-friendly interface.
- Predict the student's performance group.
- Display recommendations based on the predicted group.
- Visualize the distribution of scores and other statistics.

## Setup

### Prerequisites

Ensure you have Python 3.6 or higher installed. 

### Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/emekeys/cetm46_Ass.git
    cd student-performance-predictor
    ```

2. **Create a virtual environment:**

    ```sh
    python -m venv venv
    ```

3. **Activate the virtual environment:**

    - On Windows:
        ```sh
        venv\Scripts\activate
        ```
    - On macOS and Linux:
        ```sh
        source venv/bin/activate
        ```

4. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Run the Streamlit app:**

    ```sh
    streamlit run app.py
    ```

2. **Open your browser and navigate to:**

    ```
    http://localhost:8501
    ```

3. **Interact with the app:**

    - Enter the student's details.
    - Click on the "Predict Cluster" button to see the performance prediction and recommendations.
    - Visualize various statistics and comparisons.

## Dependencies

- streamlit
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib
- pickle

These are listed in the `requirements.txt` file and can be installed using the `pip install -r requirements.txt` command.

## Deployment

To deploy this app on Streamlit Community Cloud:

1. **Push your code to a GitHub repository.**
2. **Go to [Streamlit Community Cloud](https://streamlit.io/cloud) and sign in with GitHub.**
3. **Click on "New app", select your repository, branch, and the `app.py` file.**
4. **Click "Deploy".**

Streamlit will automatically install the dependencies listed in your `requirements.txt` file and deploy your app.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any changes.
