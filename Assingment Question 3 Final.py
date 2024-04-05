import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

defective_folder = "C:\\Users\\User\\Downloads\\itmla3_project_dataset (1)\\CONCRETE CRACKS\\Defective"
defectless_folder = "C:\\Users\\User\\Downloads\\itmla3_project_dataset (1)\\CONCRETE CRACKS\\Defectless"

def image_enhancement(image):
    # Apply histogram equalization for contrast enhancement
    # enhanced_image = cv2.equalizeHist(image)
    return image

def feature_extraction(images):
    # Implement feature extraction
    # Example: Here, we flatten each image into a 1D array of pixel values
    features = [image.flatten() for image in images]
    return np.array(features)

def feature_normalisation(X_train, X_test):
    # Normalize features
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    return X_train_normalized, X_test_normalized

def classifier_training(X_train, y_train):
    # Train the classifier
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    return clf

def classifier_testing(clf, X_test, y_test):
    # Test the classifier
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

def def_train_test_split(X, y):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training data shape:", X_train.shape, y_train.shape)
    print("Testing data shape:", X_test.shape, y_test.shape)
    return X_train, X_test, y_train, y_test

def load_images_from_folder(folder, label):
    images = []
    labels = []
    count = 0
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
        if img is not None:
            # Apply image enhancement
            img = image_enhancement(img)
            images.append(img)
            labels.append(label)
            count = count + 1
            if count <= 5:
                # Display the first 5 images
                plt.imshow(img, cmap='gray')
                plt.title(f"Label: {label}")
                plt.show()
    print(count, "labeled as", label)
    return images, labels

# Load defective and defectless images and their labels
defective_images, defective_labels = load_images_from_folder(defective_folder, 1)
defectless_images, defectless_labels = load_images_from_folder(defectless_folder, 0)

X = np.array(defective_images + defectless_images)
y = np.array(defective_labels + defectless_labels)

# Step 1: Feature Extraction
X_features = feature_extraction(X)

# Step 2: Train-Test Split
X_train, X_test, y_train, y_test = def_train_test_split(X_features, y)

# Step 3: Feature Normalization
X_train_normalized, X_test_normalized = feature_normalisation(X_train, X_test)

# Step 4: Classifier Training
clf = classifier_training(X_train_normalized, y_train)

# Step 5: Classifier Testing
classifier_testing(clf, X_test_normalized, y_test)
