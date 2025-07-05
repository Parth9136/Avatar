# import time
# import random
# import requests

# API_URL = 'http://localhost:5000/predict'  # Update this if your Flask server uses a different port

# # Generate a list of 50 records (each with scalar values)
# def generate_fake_input():
#     sequence_length = 50
#     data = []

#     for i in range(sequence_length):
#         data.append({
#             'Timestamp(ms)': int(time.time() * 1000) + i * 20,
#             'ECG_Value': round(random.uniform(0.7, 1.2), 2),
#             'AccelX': round(random.uniform(-1, 1), 2),
#             'AccelY': round(random.uniform(-1, 1), 2),
#             'AccelZ': round(random.uniform(9.5, 10.5), 2),
#             'GyroX': round(random.uniform(-30, 30), 2),
#             'GyroY': round(random.uniform(-30, 30), 2),
#             'GyroZ': round(random.uniform(-30, 30), 2),
#             'TempC': round(random.uniform(36.5, 38.0), 2),
#             'IR': random.randint(300, 800),
#             'BPM': random.randint(60, 120),
#             'Avg_BPM': random.randint(60, 120),
#         })

#     return data

# # Send data to Flask API every 5 seconds
# def main():
#     print("📡 Starting simulator — sending data to Flask API every 5 seconds...\n")
#     while True:
#         input_data = generate_fake_input()

#         try:
#             response = requests.post(API_URL, json=input_data)
#             if response.status_code == 200:
#                 result = response.json()
#                 print(f"✅ Response: Emotion = {result['emotion']}, Confidence = {result['confidence']:.2f}")
#             else:
#                 print(f"❌ Server error: {response.status_code} — {response.text}")
#         except Exception as e:
#             print(f"❌ Exception occurred: {e}")

#         time.sleep(5)

# if __name__ == "__main__":
#     main()
