# **Chest X-Ray Classification Using CNN and Pretrained Models**

## **Project Overview**

This project employs **Convolutional Neural Networks (CNN)** and pretrained models (such as ResNet, VGG, or EfficientNet) to classify chest X-ray images into conditions like _Normal_, _Viral Pneumonia_, or _Bacterial Pneumonia_ . A **Streamlit** application has been developed for interactive user access to the model predictions. The dataset used is the [CoronaHack - Chest X-Ray Dataset](https://www.kaggle.com/datasets/praveengovi/coronahack-chest-xraydataset).

## **Dataset**

The dataset consists of:

- **Images**: X-ray scans of patients with various conditions.
- **Metadata (CSV)**: Patient information, diagnosis labels, and other attributes.

### **Dataset Files**

1. **Chest_xray_Corona_Metadata.csv**: Metadata with patient details and labels.
2. **Image Folders**:
   - `Dataset/train/`: Training images.
   - `Dataset/test/`: Test images.

## **Tech Stack**

- **Framework**: TensorFlow/Keras
- **Languages**: Python
- **Libraries**:
  - TensorFlow
  - Keras
  - NumPy
  - Pandas
  - Matplotlib
  - Seaborn
  - Streamlit
- **Deployment Tool**: Streamlit

## **Steps to Run the Project**

### **1. Prerequisites**

- Python 3.x installed.
- Install required Python packages:
  ```bash
  pip install tensorflow keras numpy pandas matplotlib seaborn streamlit
  ```

## **Requirements**

To set up and run the project, ensure the following software and libraries are installed:

### Prerequisites:

1. Python 3.8 or higher
2. Virtual Environment (recommended)
3. Git

## Usage

### Step 1: Clone the Repository

Clone the GitHub repository to your local machine:

```bash
git clone <repository-url>
cd medical_ai
```

### Step 2: Set Up Virtual Environment

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Python Libraries:

Install the required libraries using `pip`:

```bash
pip install -r requirements.txt
```

### Step 3: Run the Streamlit App

Run the Streamlit application locally:

```bash
streamlit run app.py
```

### Step 4: Interact with the App

Select model choice for prediction
Upload X-ray images.
View the model's accuracy displayed in the app.
