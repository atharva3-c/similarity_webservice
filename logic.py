from fastapi import FastAPI, HTTPException, File, UploadFile
import tensorflow as tf
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import psycopg2
import tempfile
import os
from dotenv import load_dotenv
from io import BytesIO
from urllib.parse import urlparse

# Disable GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

load_dotenv()


app = FastAPI()

# Using MobileNetV2 instead of VGG16
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)  # Exclude fully connected layers
model = tf.keras.models.Model(inputs=base_model.input, outputs=base_model.output)

# Use MobileNetV2's preprocess_input function
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

def get_db_connection():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise Exception("DATABASE_URL not set in environment variables")

    # Parse the database URL
    url_parts = urlparse(db_url)
    
    # Connect using parsed components
    conn = psycopg2.connect(
        dbname=url_parts.path[1:],  # Remove leading slash
        user=url_parts.username,
        password=url_parts.password,
        host=url_parts.hostname,
        port=url_parts.port
    )
    return conn

def extract_frames_from_bytes(video_bytes, frame_interval=1):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_bytes)
        temp_video_path = temp_video.name

    cap = cv2.VideoCapture(temp_video_path)
    frames = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame = cv2.resize(frame, (224, 224))  # MobileNetV2's input size
            frames.append(frame)
        count += 1

    cap.release()
    os.remove(temp_video_path)
    return np.array(frames)

def resize_feature_vector(feature_vector, target_size=4096):
    # If feature vector is larger than target size, reduce it by taking the first `target_size` elements
    if feature_vector.size > target_size:
        return feature_vector[:target_size]
    # If smaller, pad the feature vector with zeros
    elif feature_vector.size < target_size:
        return np.pad(feature_vector, (0, target_size - feature_vector.size), 'constant')
    return feature_vector

def extract_feature_vector(video_bytes, frame_interval=1, target_size=4096):
    frames = extract_frames_from_bytes(video_bytes, frame_interval)
    features = []
    for frame in frames:
        frame = np.expand_dims(frame, axis=0)
        frame = preprocess_input(frame)
        feature = model.predict(frame)
        feature_flattened = feature.flatten()
        feature_resized = resize_feature_vector(feature_flattened, target_size)
        features.append(feature_resized)
    return np.array(features)

def get_all_feature_vectors():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT video_id, feature_vector FROM video_features")
    results = cursor.fetchall()
    cursor.close()
    conn.close()

    feature_vectors = []
    for video_id, feature_vector in results:
        feature_vector_array = np.array(feature_vector)
        num_frames = feature_vector_array.size // 4096  # Calculate the number of frames based on vector size
        print(f"Video ID: {video_id}, Feature Vector Shape: {feature_vector_array.shape}, Number of Frames: {num_frames}")

        # Reshape dynamically based on the number of frames
        reshaped_vector = feature_vector_array.reshape(num_frames, 4096)
        feature_vectors.append((video_id, reshaped_vector))

    return feature_vectors

def insert_new_feature_vector(feature_matrix):
    flattened_vector = feature_matrix.flatten().tolist()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO video_features (feature_vector)
        VALUES (%s)
        RETURNING video_id
        """,
        (flattened_vector,)
    )
    video_id = cursor.fetchone()[0]
    conn.commit()
    cursor.close()
    conn.close()
    return video_id

def compare_with_fixed_vector(Fv_fixed, threshold=0.5):
    all_feature_vectors = get_all_feature_vectors()
    similarities = []
    for video_id, feature_vector in all_feature_vectors:
        frame_similarities = [cosine_similarity([f_fixed], [f])[0][0] for f_fixed, f in zip(Fv_fixed, feature_vector)]
        average_similarity = np.mean(frame_similarities)
        similarities.append((video_id, average_similarity))

    max_video_id, max_similarity = max(similarities, key=lambda x: x[1])
    if max_similarity < threshold:
        new_video_id = insert_new_feature_vector(Fv_fixed)
        return new_video_id, max_similarity
    else:
        return max_video_id, max_similarity

@app.post("/compare-video-bytes/")
async def compare_video(file: UploadFile = File(...)):
    try:
        # Read video bytes from the uploaded file
        video_bytes = await file.read()

        # Process the video bytes to extract features
        features_fixed = extract_feature_vector(video_bytes, frame_interval=30)
        video_id, max_similarity = compare_with_fixed_vector(features_fixed)

        return {"video_id": video_id, "max_similarity": max_similarity}

    except Exception as e:
        # More detailed error logging
        import traceback
        print("Detailed traceback:", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error. Please check logs.")
