function showModal(modalId) {
    var modal = document.getElementById(modalId);
    modal.style.display = 'block';
    if (modalId === 'generateSlidePointsModal' || modalId === 'openUploadedFilesModal' || modalId === 'openGeneratedSlidePointsModal' || modalId === 'generateSlidesModal') {
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
        
        statusMessage.textContent = data.message || 'Processing...';
        progressBar.value = data.progress || 0;
        
        statusMessage.style.display = 'none';
        statusMessage.offsetHeight;
        statusMessage.style.display = 'block';
        
        if (data.stage === 'completed') {
            eventSource.close();
            alert('Processing completed successfully!');
            document.getElementById('progressSection').style.display = 'none';
            // Auto-download generated slides if available
            if (data.downloaded_files && Array.isArray(data.downloaded_files)) {
                data.downloaded_files.forEach(function(filename) {
                    const link = document.createElement('a');
                    link.href = `/download_slide/${encodeURIComponent(filename)}`;
                    link.download = filename;
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                });
            }
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
    let endpoint;
    if (modalId === 'generateSlidePointsModal' || modalId === 'openUploadedFilesModal') {
        endpoint = '/get_content_files';
    } else if (modalId === 'openGeneratedSlidePointsModal') {
        endpoint = '/get_generated_slide_points';
    } else if (modalId === 'generateSlidesModal') {
        endpoint = '/generate_slides_pptx';
    }

    fetch(endpoint)
        .then(response => response.json())
        .then(data => {
            let fileSelect;
            if (modalId === 'generateSlidePointsModal') {
                fileSelect = document.getElementById('sp_filenames');
            } else if (modalId === 'openUploadedFilesModal') {
                fileSelect = document.getElementById('of_filenames');
            } else if (modalId === 'openGeneratedSlidePointsModal') {
                fileSelect = document.getElementById('ogsp_filenames');
            } else if (modalId === 'generateSlidesModal') {
                fileSelect = document.getElementById('gs_filenames');
            }
            fileSelect.innerHTML = '';
            if (data.files && Array.isArray(data.files)) {
                data.files.forEach(file => {
                    const option = document.createElement('option');
                    option.value = file;
                    option.text = file;
                    fileSelect.appendChild(option);
                });
            }
            if (modalId === 'generateSlidePointsModal') {
                filterFiles();
            } else if (modalId === 'openUploadedFilesModal') {
                filterUploadedFiles();
            } else if (modalId === 'openGeneratedSlidePointsModal') {
                filterGeneratedSlidePoints();
            } else if (modalId === 'generateSlidesModal') {
                filterGeneratedFiles();
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

function filterGeneratedSlidePoints() {
    const grade = document.getElementById('ogsp_grade').value.toLowerCase();
    const course = document.getElementById('ogsp_course').value.toLowerCase();
    const section = document.getElementById('ogsp_section').value.toLowerCase();
    const language = document.getElementById('ogsp_language').value.toLowerCase();
    const country = document.getElementById('ogsp_country').value.toLowerCase();
    const fileSelect = document.getElementById('ogsp_filenames');
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
        .then(response => {
            if (!response.ok) throw new Error('Failed to fetch text file');
            return response.text();
        })
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

function openGeneratedSlidePoints() {
    const fileSelect = document.getElementById('ogsp_filenames');
    const selectedFile = fileSelect.value;

    if (!selectedFile) {
        alert('Please select a slide points file.');
        return;
    }

    fetch(`/view_pdf/${encodeURIComponent(selectedFile)}`)
        .then(response => {
            if (!response.ok) throw new Error('Failed to fetch PDF');
            return response.blob();
        })
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const newWindow = window.open(url, '_blank');
            if (!newWindow) {
                alert('Failed to open PDF. Please allow pop-ups.');
            } else {
                newWindow.document.title = selectedFile;
            }
            window.URL.revokeObjectURL(url);
            hideModal('openGeneratedSlidePointsModal');
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while opening the file.');
        });
}

function downloadGeneratedSlidePoints() {
    const fileSelect = document.getElementById('ogsp_filenames');
    const selectedFile = fileSelect.value;

    if (!selectedFile) {
        alert('Please select a slide points file.');
        return;
    }

    hideModal('openGeneratedSlidePointsModal');
    window.location.href = `/download_generated_slide_points/${encodeURIComponent(selectedFile)}`;
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
                startProgress(data.job_id);
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

function generateSlides() {
    var form = document.getElementById('generateSlidesForm');
    var formData = new FormData(form);
    // No need to manually set theme, as it is included in the form
    form.reset();
    hideModal('slideParamsModal');
    selectedFiles.forEach(filename => {
        formData.set('filename', filename);
        fetch('/generate_slides_pptx', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert('Presentation generation started for ' + filename);
                startProgress(data.job_id);
            } else {
                alert('Generation failed for ' + filename + ': ' + data.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred during presentation generation for ' + filename);
        });
    });
}

let selectedFiles = [];

function filterGeneratedFiles() {
    const grade = document.getElementById('gs_grade').value.toLowerCase();
    const course = document.getElementById('gs_course').value.toLowerCase();
    const section = document.getElementById('gs_section').value.toLowerCase();
    const language = document.getElementById('gs_language').value.toLowerCase();
    const country = document.getElementById('gs_country').value.toLowerCase();
    const fileSelect = document.getElementById('gs_filenames');
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

function openSlideParamsModal() {
    const fileSelect = document.getElementById('gs_filenames');
    selectedFiles = Array.from(fileSelect.selectedOptions).map(option => option.value);

    if (selectedFiles.length === 0) {
        alert('Please select at least one slide points file.');
        return;
    }

    hideModal('generateSlidesModal');
    showModal('slideParamsModal');
}

function generateSlides() {
    var form = document.getElementById('generateSlidesForm');
    var formData = new FormData(form);

    form.reset();
    hideModal('slideParamsModal');

    selectedFiles.forEach(filename => {
        formData.set('filename', filename);
        fetch('/generate_slides_pptx', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert('Presentation generation started for ' + filename);
                startProgress(data.job_id);
            } else {
                alert('Generation failed for ' + filename + ': ' + data.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred during presentation generation for ' + filename);
        });
    });
}

function fetchGeneratedSlidesFiles() {
    fetch('/generate_slides_pptx', { method: 'GET' })
        .then(response => response.json())
        .then(data => {
            const fileSelect = document.getElementById('gs_filenames');
            fileSelect.innerHTML = '';
            if (data.files && Array.isArray(data.files)) {
                data.files.forEach(file => {
                    const option = document.createElement('option');
                    option.value = file;
                    option.text = file;
                    fileSelect.appendChild(option);
                });
            }
            filterGeneratedFiles();
        })
        .catch(error => {
            console.error('Error fetching generated slides files:', error);
            alert('Error fetching generated slides files.');
        });
}
