import json

import requests


def send_image_url(image_url: str):
    # Define the server URL
    server_url = "http://127.0.0.1:5000/predict"  # Replace with your server's URL if different

    # Prepare the payload
    payload = {
        "image_path": image_url
    }

    try:
        # Send the POST request
        response = requests.post(server_url, json=payload)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse and print the JSON response
            prediction = response.json()
            print(f"Prediction: {prediction}")
        else:
            print(f"Failed to get a valid response. Status code: {response.status_code}")
            print(f"Response: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    image_url = "https://firebasestorage.googleapis.com/v0/b/auth-react-app-89780.appspot.com/o/1713685642011esp32.jpg?alt=media&token=8b16df78-ad1f-4172-be6a-ab909d27f1fd"
    send_image_url(image_url)
