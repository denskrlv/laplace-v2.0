import pandas as pd
import requests
import os


def download_data():
    """
    Downloads the UCI Adult dataset into the current directory.
    """
    train_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    test_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    print("Downloading train data to adult.data...")
    try:
        response = requests.get(train_url, headers=headers)
        response.raise_for_status()
        with open('adult.data', 'w') as f:
            f.write(response.text)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading train data: {e}")
        return

    print("Downloading test data to adult.test...")
    try:
        response = requests.get(test_url, headers=headers)
        response.raise_for_status()
        with open('adult.test', 'w') as f:
            f.write(response.text)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading test data: {e}")
        return

    print("\nDataset downloaded successfully.")


if __name__ == '__main__':
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Change the current working directory to the script's directory
    os.chdir(script_dir)
    print(f"Working directory changed to: {os.getcwd()}")

    download_data()