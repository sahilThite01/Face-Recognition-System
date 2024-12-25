
from flask import Flask, render_template, request, redirect, url_for,send_from_directory,jsonify
import os
import subprocess
import pickle
from werkzeug.utils import secure_filename


app = Flask(__name__)

# Create the 'dataset' directory if it doesn't exist
if not os.path.exists("dataset"):
    os.makedirs("dataset")


@app.route("/delete_folders", methods=["POST"])
def delete_folders():
    # Delete the folders and files
    try:
        import shutil
        shutil.rmtree("dataset")
        shutil.rmtree("uploads")
        os.remove("encodings.pickle")
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Function to execute encode_faces.py if dataset folder is empty
def check_dataset_folder():
    if not os.path.exists("dataset"):
        # If it doesn't exist, create it
        os.makedirs("dataset")
        print("Dataset folder created.")
        #if dataset is empty, open encodings.pickle and initialize it
    if not os.listdir("dataset"):
        knownEncodings = []
        knownNames = []
        data = {"encodings": knownEncodings, "names": knownNames}
        with open("encodings.pickle", "wb") as f:
            f.write(pickle.dumps(data))

# Route for the main page
@app.route("/", methods=["GET", "POST"])
def index():
    # Check if the request method is POST and the clear button is clicked
    if request.method == "POST" and request.form.get("clear_button"):
        # Delete the dataset folder, uploads folder, and encodings.pickle file
        delete_folder("dataset")
        delete_folder("uploads")
        delete_file("encodings.pickle")

    return render_template("index.html", unique_videos=None)


# Function to delete a folder
def delete_folder(folder_name):
    folder_path = os.path.join(os.getcwd(), folder_name)
    if os.path.exists(folder_path):
        subprocess.run(["rm", "-rf", folder_path])

# Function to delete a file
def delete_file(file_name):
    file_path = os.path.join(os.getcwd(), file_name)
    if os.path.exists(file_path):
        os.remove(file_path)

    


@app.route('/serve_dataset_video/<path:filename>')
def serve_dataset_video(filename):
    video_directory = 'dataset'
    return send_from_directory(video_directory, filename, mimetype='video/mp4')



@app.route("/upload", methods=["POST"])
def upload_folder():
    if request.method == "POST":
        # Get the uploaded folder
        uploaded_folder = request.files.getlist("folder")
        print(uploaded_folder)
        
        # Save the uploaded folder
        folder_path = os.path.join(app.config["UPLOAD_FOLDER"], "uploaded_folder")
        os.makedirs(folder_path, exist_ok=True)

        for file in uploaded_folder:
            if file:
                # Secure the filename to prevent any malicious characters
                filename = secure_filename(file.filename)
                file_path = os.path.join(folder_path, filename)
                print("Uploaded file path:", file_path)
                file.save(file_path)

        unique_videos = process_uploaded_folder(folder_path)
        
        # Return a JSON response with processing status and unique videos
        return jsonify({'processing_complete': True, 'unique_videos': unique_videos})
    
    # If the request method is not POST, return an empty JSON response
    return jsonify({})

# Function to process videos in the uploaded folder
def process_uploaded_folder(folder_path):
    check_dataset_folder()
    subprocess.run(["python", "match_faces.py", "--encodings", "encodings.pickle", "--input", folder_path])
    dataset_folder = "dataset"
    unique_videos = os.listdir(dataset_folder)
    return unique_videos



if __name__ == "__main__":
    app.config["UPLOAD_FOLDER"] = "uploads"
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    app.run(debug=True)
