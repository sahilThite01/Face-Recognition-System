 

// Function to clear the unique videos list
function clearUniqueVideosList() {
    const uniqueVideosList = document.getElementById('videoContainer');
    uniqueVideosList.innerHTML = ''; // Clear the content of the list
}

// Function to reset the file input field
function resetFileInput() {
    const fileInput = document.getElementById('videoFiles');
    fileInput.value = ''; // Clear the value of the file input field
}

// clear button 
window.onload = function() {
    // Add event listener to the clear button
    const clearButton = document.getElementById('clearButton');
    clearButton.addEventListener('click', function() {
        clearUniqueVideosList(); // Clear the unique videos list
        resetFileInput(); // Reset the file input field
        deleteFolders(); // Delete folders and files
        window.location.href='/';
    });

// Function to delete folders and files
function deleteFolders() {
    fetch('/delete_folders', {
        method: 'POST',
        body: JSON.stringify({}),
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => {
        if (response.ok) {
            console.log('Folders and files deleted successfully.');
        } else {
            console.error('Failed to delete folders and files.');
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
}
};

// JavaScript to handle form submission and show loading spinner
document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const uploadButton= document.getElementById('uploadButton');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const videoContainer = document.getElementById('videoContainer');

    uploadForm.addEventListener('submit', function(event) {
        event.preventDefault();
        loadingSpinner.style.display = 'block'; // Show the spinner when form is submitted
        videoContainer.style.display = 'none'; // Hide the video container until processing is complete

        // Submit the form asynchronously using fetch API
        fetch('/upload', {
            method: 'POST',
            body: new FormData(this) // FormData object containing the form data
        })
        .then(response => {
            console.log('Response:', response);
            return response.json();
        })
        .then(data => {
            console.log('Received data:', data); // Log the received data to the console

            if (data.processing_complete && data.unique_videos) {
                // Processing is complete and result videos are available
                // Hide the spinner
                loadingSpinner.style.display = 'none';
                // Update the video container with the result videos
                updateVideoContainer(data.unique_videos);
                // Show the container with the result videos
                videoContainer.style.display = 'block';
            } else {
                console.error('Processing failed or no videos found.');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            // Hide the spinner in case of error
            loadingSpinner.style.display = 'none';
            // Handle errors if any
        });
    });
});


function updateVideoContainer(uniqueVideos) {
    const videoListElement = document.getElementById('videoContainer');
    videoListElement.innerHTML = ''; // Clear the existing content

    if (uniqueVideos && uniqueVideos.length > 0) {
        const ulElement = document.createElement('ul');
        const h3Element = document.createElement('h3');
        h3Element.textContent = 'Unique Faces Found:';
        ulElement.appendChild(h3Element);

        uniqueVideos.forEach(video => {
            const liElement = document.createElement('li');
            const videoElement = document.createElement('video');
            videoElement.controls = true;

            const sourceElement = document.createElement('source');
            sourceElement.src = '/serve_dataset_video/' + video;
            sourceElement.type = 'video/mp4';

            videoElement.appendChild(sourceElement);
            liElement.appendChild(videoElement);
            ulElement.appendChild(liElement);
        });

        videoListElement.appendChild(ulElement);
    } else {
        const h3Element = document.createElement('h3');
        h3Element.textContent = 'No unique faces found.';
        videoListElement.appendChild(h3Element);
    }
}



// code to disabled the upload button till user select any file
function checkFiles() {
    const fileInput = document.getElementById('videoFiles');
    const uploadButton = document.getElementById('uploadButton');
// We check if the length of this array is greater than 0. If it is, it means the user has selected one or more files, so we enable the upload button. Otherwise, we disable it.
    if (fileInput.files.length > 0) {
        uploadButton.disabled = false;
    } else {
        uploadButton.disabled = true;
    }
}


