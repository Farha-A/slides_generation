function showModal(modalId) {
    var modal = document.getElementById(modalId);
    modal.style.display = 'block';
    if (modalId === 'generateSlidePointsModal') {
        fetchFiles();
    }
}

function hideModal(modalId) {
    var modal = document.getElementById(modalId);
    modal.style.display = 'none';
}

function startProgress(jobId) {
    document.getElementById('progressSection').style.display = 'block';
    const eventSource = new EventSource(`/progress/${jobId}`);
    eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        document.getElementById('statusMessage').textContent = data.message;
        document.getElementById('progressBar').value = data.progress;
        if (data.stage === 'completed') {
            eventSource.close();
            alert('Processing completed successfully!');
            document.getElementById('progressSection').style.display = 'none';
        } else if (data.stage === 'error') {
            eventSource.close();
            alert('Error: ' + data.message);
            document.getElementById('progressSection').style.display = 'none';
        }
    };
    eventSource.onerror = function() {
        eventSource.close();
        document.getElementById('progressSection').style.display = 'none';
        alert('Progress tracking failed. Please try again.');
    };
}

function uploadFile() {
    var form = document.getElementById('uploadForm');
    var formData = new FormData(form);

    form.reset();

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('File uploaded successfully. Processing started.');
            hideModal('uploadModal');
            startProgress(data.job_id);
        } else {
            alert('Upload failed: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred during upload.');
    });
}

function fetchFiles() {
    fetch('/get_content_files')
        .then(response => response.json())
        .then(data => {
            const fileSelect = document.getElementById('sp_filenames');
            fileSelect.innerHTML = '';
            data.files.forEach(file => {
                const option = document.createElement('option');
                option.value = file;
                option.text = file;
                fileSelect.appendChild(option);
            });
            filterFiles();
        })
        .catch(error => {
            console.error('Error fetching files:', error);
            alert('Error fetching content files.');
        });
}

function filterFiles() {
    const grade = document.getElementById('sp_grade').value.toLowerCase();
    const course = document.getElementById('sp_course').value.toLowerCase();
    const section = document.getElementById('sp_section').value.toLowerCase();
    const language = document.getElementById('sp_language').value.toLowerCase();
    const country = document.getElementById('sp_country').value.toLowerCase();
    const fileSelect = document.getElementById('sp_filenames');
    const options = fileSelect.options;

    for (let i = 0; i < options.length; i++) {
        const file = options[i].value.toLowerCase();
        const show =
            (grade === '' || file.includes(grade)) &&
            (course === '' || file.includes(course)) &&
            (section === '' || file.includes(section)) &&
            (language === '' || file.includes(language)) &&
            (country === '' || file.includes(country));
        options[i].style.display = show ? '' : 'none';
    }
}

function generateSlidePoints() {
    var form = document.getElementById('generateSlidePointsForm');
    var formData = new FormData(form);
    var selectedFiles = Array.from(document.getElementById('sp_filenames').selectedOptions).map(option => option.value);

    if (selectedFiles.length === 0) {
        alert('Please select at least one content file.');
        return;
    }

    form.reset();
    hideModal('generateSlidePointsModal');

    selectedFiles.forEach(filename => {
        formData.set('filename', filename);
        const job_id = `gen_${Date.now()}_${filename}`;
        startProgress(job_id);
        fetch('/generate_slides', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert('Slide points generated successfully for ' + filename);
            } else {
                alert('Generation failed for ' + filename + ': ' + data.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred during slide points generation for ' + filename);
        });
    });
}