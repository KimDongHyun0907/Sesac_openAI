var dropArea = document.getElementById('drop-area');

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

// 드래그 중일 때 스타일 변경
['dragenter', 'dragover'].forEach(eventName => {
    dropArea.addEventListener(eventName, function(e) {
        dropArea.classList.add('dragover');
        document.querySelector('.message').innerText = '파일을 여기에 드롭하세요!';
        preventDefaults(e); // 기본 동작 중지
    }, false);
});

// 드래그를 떠날 때 스타일 변경 및 메시지 출력
dropArea.addEventListener('dragleave', function() {
    dropArea.classList.remove('dragover');
    document.querySelector('.message').innerText = '업로드할 이미지를 드래그하거나 파일 추가 버튼을 클릭하여 이미지를 업로드 하세요.';
}, false);

dropArea.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
    var dt = e.dataTransfer;
    var files = dt.files;

    handleFiles(files);
}

document.getElementById('file-input').addEventListener('change', function() {
    handleFiles(this.files);
});

function handleFiles(files) {
    var file = files[0];

    if (file) {
        uploadFile(file);
    }
}

function uploadFile(file) {
    var formData = new FormData();
    formData.append('file', file);

    fetch('/prediction', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.redirect_url) {
            window.location.href = data.redirect_url
        }
    })
    .catch(error => console.error('Error:', error));
}