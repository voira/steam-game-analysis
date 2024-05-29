import pandas as pd
import torch
import cv2
import torchvision.transforms as transforms
from collections import Counter
from model import CNNModel
from os.path import exists

# path of the images
file_path="C:/Users/nicol/OneDrive - KU Leuven/Desktop/python/Advanced Analytics in Business/2/test"
# the computation device
device = ("cuda")

# list containing all the class labels
labels = [
    "Mixed","Mostly Negative","Mostly Positive","Negative","Overwhelmingly Positive","Positive","Very Positive","Very Negative"
    ]
# herarchy of the labels
hierarchy={"Mixed":6,
           "Mostly Negative":3,
           "Mostly Positive":5,
           "Negative":2,
           "Overwhelmingly Positive":4,
           "Positive":8,
           "Very Positive":7,
           "Very Negative":1}

# initialize the model and load the trained weights
model = CNNModel(8).to(device)
checkpoint = torch.load('C:/Users/nicol/OneDrive - KU Leuven/Desktop/python/Advanced Analytics in Business/2/old model/model 2/model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load the JSON data into a python dictionary
test_data = pd.read_json("C:/Users/nicol/OneDrive - KU Leuven/Desktop/python/Advanced Analytics in Business/2/test_data.json")
# Clean out the games that have no reviews
test_df = test_data.dropna(subset=["sentiment"])
# Initialize the "prediction" column with None
test_df["predictions"] = None

# define preprocess transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


def check_higher(prediction_list):
    prediction_counts = Counter(prediction_list)
    max_count = max(prediction_counts.values())
    most_common_predictions = [pred for pred, count in prediction_counts.items() if count == max_count]
    if len(most_common_predictions) == 1:
        return most_common_predictions[0]
    else:
        sorted_predictions = sorted(most_common_predictions, key=lambda x: hierarchy.get(x, 0), reverse=True)
        return sorted_predictions[0]


def preprocess_image(image_path):
    # Load image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Apply the transform
    image = transform(image)
    # Add batch dimension
    image = image.unsqueeze(0).to(device)
    return image


def group_check(game):
    df_expanded=game.explode("screenshots", ignore_index=True)
    predictions=[]
    for i in range(len(df_expanded["screenshots"])):
        screenshot_filename = df_expanded["screenshots"][i]
        sentiment = df_expanded["sentiment"][i]
        image_path = f"{file_path}/{sentiment}/{screenshot_filename}"
        # Preprocess the image
        if exists(image_path):
            image = preprocess_image(image_path)
            # Make prediction
            with torch.no_grad():
                outputs = model(image.to(device))
            output_label = torch.topk(outputs, 1)
            pred_class = labels[int(output_label.indices)]
            predictions.append(pred_class)
    return check_higher(predictions)


def main():
    test_correct = 0
    for index, row in test_df.iterrows():
        test_row=test_df.loc[[index]].copy()
        prediction=group_check(test_row)
        test_df.at[index, "predictions"]=prediction
        if test_df["predictions"][index]==test_df["sentiment"][index]:
            test_correct += 1
    test_df_final = test_df.dropna(subset=["predictions"])        
    print("Accuracy: ",test_correct/len(test_df_final))
    test_df_final.to_json("C:/Users/nicol/OneDrive - KU Leuven/Desktop/python/Advanced Analytics in Business/2/pred_test_data.json")


if __name__=="__main__":
    main()