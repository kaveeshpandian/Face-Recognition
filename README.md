# Face Detection and Recognition Using CNN and Random Forest with HOG Features

This project implements two machine learning models—Convolutional Neural Networks (CNN) and Random Forest with Histogram of Oriented Gradients (HOG) features—for face detection and recognition. The aim is to explore and compare the effectiveness of both models in identifying and classifying faces from a given dataset.

## Project Overview

The project is structured to provide a comprehensive solution for face detection and recognition, involving the following key steps:

1. **Data Preprocessing**: Images are resized and converted into a standard format suitable for further processing. HOG features are extracted for the Random Forest model.
2. **Feature Extraction**:
    - **CNN**: Uses the inherent feature extraction capability of convolutional layers.
    - **Random Forest**: Relies on pre-extracted HOG features to train the model.
3. **Model Training**:
    - **CNN**: A deep learning model trained on the image dataset to learn and recognize face patterns.
    - **Random Forest**: An ensemble learning method trained on the HOG features extracted from the images.
4. **Model Evaluation**: Both models are evaluated on metrics such as accuracy, precision, recall, F1-score, and confusion matrices to determine their performance.

## Installation and Setup

### Prerequisites

- Python 3.x
- Required libraries: `numpy`, `opencv-python`, `scikit-learn`, `scikit-image`, `matplotlib`, `seaborn`

### Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/face-detection-recognition.git
    cd face-detection-recognition
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the Models

1. Place your training images in the `Final Training Images` folder and testing images in the `Final Testing Images` folder.
2. Run the training script:
    ```bash
    python train_cnn.py
    python train_rf.py
    ```
3. The trained models will be saved in the `models` directory.

### Evaluating the Models

1. Use the evaluation scripts to test the models:
    ```bash
    python evaluate_cnn.py
    python evaluate_rf.py
    ```

### Visualizing Results

1. Confusion Matrix:
    - CNN: `python plot_confusion_matrix_cnn.py`
    - Random Forest: `python plot_confusion_matrix_rf.py`
2. Classification Report:
    - Both models provide a detailed classification report outlining precision, recall, and F1-score.

## Results and Performance Analysis

- **CNN Model**: Demonstrates robust feature extraction and high accuracy, particularly effective in recognizing faces in controlled environments.
- **Random Forest with HOG Features**: Offers competitive performance, especially in scenarios with limited computational resources, by reducing the dimensionality through HOG.

Comparative analysis through confusion matrices and classification reports shows the trade-offs between deep learning and traditional machine learning methods in terms of accuracy, computational efficiency, and generalization.

## Conclusion

This project provides insights into the application of CNN and Random Forest with HOG features for face detection and recognition. It highlights the strengths and limitations of both approaches, offering a solid foundation for further research and development in facial recognition systems.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

Special thanks to the open-source community and the contributors of the libraries used in this project.

---

This content should serve as a clear and informative README for your GitHub repository, explaining the project’s goals, methodology, and how to use it.
