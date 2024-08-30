import requests
import csv
import joblib
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL")
authorization_token = os.getenv("AUTHORIZATION_TOKEN")
headers = {"Authorization": authorization_token}
loaded_model = joblib.load('decision_tree_model.joblib')

THRESHOLD = 0.5
CSV_FILE = 'datasets.csv'


def pred_to_csv(data, file_name):
    """Append data to a CSV file."""
    try:
        with open(file_name, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(data)
        return True
    except Exception as e:
        print(f"Error occurred while appending data to {file_name}: {e}")
        return False



def prepare_data(data):
    """Prepare data for CSV writing."""
    try:
        max_label = None
        max_score = 0.0
        positive_score = 0
        negative_score = 0
        neutral_score = 0

        for item in data:
            if item['label'] == 'positive':
                positive_score = item['score']
            elif item['label'] == 'negative':
                negative_score = item['score']
            elif item['label'] == 'neutral':
                neutral_score = item['score']
        
        for item in data:
            if item['score'] > max_score:
                max_label = item['label']
                max_score = item['score']
                
        result = [positive_score, negative_score, neutral_score, max_label]
        return result

    except Exception as e:
        # Handle the exception, e.g., print an error message and return None
        print(f"Error occurred while preparing data: {e}")
        return None



def query(payload):
    """Make a POST request to the API."""
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for 4xx and 5xx status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        # Handle request exceptions (e.g., network errors, timeout)
        print(f"Request to the API failed: {e}")
        return None
    except ValueError as e:
        # Handle JSON decoding errors
        print(f"Error decoding JSON response: {e}")
        return None


def collect_data(input):
    """Collect data from the API and write to CSV."""
    try:
        output = query({"inputs": input})
        if output is not None and output:  # Check if output is not None and not empty
            data = prepare_data(output[0])
            if data is not None:  # Check if data is not None
                above_threshold = any(isinstance(item, (int, float)) and item > THRESHOLD for item in data)
                if above_threshold:
                    pred_to_csv(data, CSV_FILE)
                else:
                    print("Did not exceed the threshold.")
            else:
                print("Data preparation failed. CSV file not updated.")
        else:
            print("Empty or invalid output from the API. CSV file not updated.")
    except Exception as e:
        print(f"An error occurred: {e}. CSV file not updated.")



def reformat_scores(original_list):
    """Reformat scores from the API response."""
    try:
        reformatted_dict = {
            'positive_score': 0.0,
            'negative_score': 0.0,
            'neutral_score': 0.0
        }

        for item in original_list:
            if 'label' in item and 'score' in item:
                if item['label'] == 'positive':
                    reformatted_dict['positive_score'] = item['score']
                elif item['label'] == 'neutral':
                    reformatted_dict['neutral_score'] = item['score']
                elif item['label'] == 'negative':
                    reformatted_dict['negative_score'] = item['score']
            else:
                raise ValueError("Invalid item format in the original list")

        final_list = [reformatted_dict]
        return final_list
    except Exception as e:
        print(f"An error occurred while reformatting scores: {e}")
        return []


def sentiment_analysis(input):
    """Perform sentiment analysis using the loaded model."""
    try:
        output = query({"inputs": input})
        if output is not None and output:  # Check if output is not None and not empty
            data = reformat_scores(output[0])
            if data:  # Check if data is not empty
                data = np.array([[sample['positive_score'], sample['negative_score'], sample['neutral_score']] for sample in data])
                new_predictions = loaded_model.predict(data)
                string = new_predictions[0]
                return string
            else:
                print("Reformatted scores are empty. Sentiment analysis failed.")
                return None
        else:
            print("Empty or invalid output from the API. Sentiment analysis failed.")
            return None
    except Exception as e:
        print(f"An error occurred during sentiment analysis: {e}")
        return None



