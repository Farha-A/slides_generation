function showModal(modalId) {
    var modal = document.getElementById(modalId);
    modal.style.display = 'block';
    if (modalId === 'generateSlidePointsModal' || modalId === 'openUploadedFilesModal') {
        fetchFiles(modalId);
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
        const statusMessage = document.getElementById('statusMessage');
        const progressBar = document.getElementById('progressBar');
        
        // Update UI elements
        statusMessage.textContent = data.message || 'Processing...';
        progressBar.value = data.progress || 0;
        
        // Force UI refresh
        statusMessage.style.display = 'none';
        statusMessage.offsetHeight; // Trigger reflow
        statusMessage.style.display = 'block';
        
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

function fetchFiles(modalId) {
    fetch('/get_content_files')
        .then(response => response.json())
        .then(data => {
            const fileSelect = modalId === 'generateSlidePointsModal' ? 
                document.getElementById('sp_filenames') : 
                document.getElementById('of_filenames');
            fileSelect.innerHTML = '';
            data.files.forEach(file => {
                const option = document.createElement('option');
                option.value = file;
                option.text = file;
                fileSelect.appendChild(option);
            });
            if (modalId === 'generateSlidePointsModal') {
                filterFiles();
            } else {
                filterUploadedFiles();
            }
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

function filterUploadedFiles() {
    const grade = document.getElementById('of_grade').value.toLowerCase();
    const course = document.getElementById('of_course').value.toLowerCase();
    const section = document.getElementById('of_section').value.toLowerCase();
    const language = document.getElementById('of_language').value.toLowerCase();
    const country = document.getElementById('of_country').value.toLowerCase();
    const fileSelect = document.getElementById('of_filenames');
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

function openUploadedFile() {
    const fileSelect = document.getElementById('of_filenames');
    const selectedFile = fileSelect.value;

    if (!selectedFile) {
        alert('Please select a content file.');
        return;
    }

    fetch(`/view_text_file/${encodeURIComponent(selectedFile)}`)
        .then(response => response.text())
        .then(data => {
            const newWindow = window.open('', '_blank');
            newWindow.document.write('<pre>' + data + '</pre>');
            newWindow.document.title = selectedFile;
            hideModal('openUploadedFilesModal');
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while opening the file.');
        });
}

function downloadUploadedFile() {
    const fileSelect = document.getElementById('of_filenames');
    const selectedFile = fileSelect.value;

    if (!selectedFile) {
        alert('Please select a content file.');
        return;
    }

    hideModal('openUploadedFilesModal');
    window.location.href = `/download_text_as_pdf/${encodeURIComponent(selectedFile)}`;
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
        fetch('/generate_slides', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert('Slide points generation started for ' + filename);
                startProgress(data.job_id); // Use server-provided job_id
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