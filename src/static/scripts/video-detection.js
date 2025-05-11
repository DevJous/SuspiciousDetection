var framesSkipToAnalyze = 3; // por defecto
var isProcessing = false;
var detecciones = ['xd'];
var filename = '';

document.addEventListener('DOMContentLoaded', function () {
    const uploadForm = document.getElementById('upload-form');
    const videoFile = document.getElementById('video-file');
    const uploadBtn = document.getElementById('upload-btn');
    const reuploadBtn = document.getElementById('reupload-btn');
    const initInfo = document.getElementById('init-info');
    const progressContainer = document.getElementById('progress-container');
    const videoContainer = document.getElementById('video-container');
    // const resultsContainer = document.getElementById('results-container');
    // const processedVideo = document.getElementById('processed-video');
    // const timestampsList = document.getElementById('timestamps-list');
    const imgProcessed = document.getElementById('img-processed-video');
    const videoPreview = document.getElementById("loaded-video");
    const ddMenuButton = document.getElementById("dropdownMenuButton");
    const ddItems = document.querySelectorAll('.dropdown-item');
    const labelWarning = document.getElementById('label-warning');
    const loadVideoInfo = document.getElementById('load-video-info');
    const viewResultsBtn = document.getElementById('view-results');
    const closeModalBtn = document.getElementById('close-btn');
    const fullVideo = document.getElementById('full-video');
    const modal = document.getElementById('myModal');

    uploadForm.addEventListener('submit', function (e) {
        e.preventDefault();
        if (!videoFile.files[0]) {
            showToast("Primero seleccione un archivo de video para continuar", "info");
            return;
        }

        const formData = new FormData();
        formData.append('video', videoFile.files[0]);

        fetch('http://localhost:5000/save-video', {
            method: 'POST',
            body: formData
        })
            .then(async res => {
                const data = await res.json();
                if (!res.ok) {
                    throw data;
                }
                goToDetection(data.filename)
                ddMenuButton.disabled = true;
                uploadBtn.disabled = true;
                videoFile.disabled = true;
                showToast("Procesando video", "info");
            })
            .catch(err => {
                if (err.message && err.message.includes('existe')) {
                    goToDetection(err.filename);
                    ddMenuButton.disabled = true;
                    uploadBtn.disabled = true;
                    videoFile.disabled = true;
                    showToast("Procesando video", "info");
                } else {
                    showToast("Error al procesar el video", "error");
                }
            });
    })

    function goToDetection(filename) {
        init = false;
        const source = new EventSource(`/stream_frames/${filename}/${framesSkipToAnalyze}`);

        source.onmessage = function (event) {
            if (event.data === "EOF") {
                source.close();
                showToast("Video procesado correctamente", "success");
                fetch('/detecciones')
                    .then(res => res.json())
                    .then(data => {
                        progressContainer.style.display = 'none';
                        detecciones = data;
                        // console.log("Detecciones:", data)
                    });
            } else {
                if (event.data && !init) {
                    videoPreview.playbackRate = 0.5; // 0.5x (lento) - 1.0x (normal) - 1.5x (rapido) - 2.0x (muy rapido)
                    reuploadBtn.style.display = 'inline-block';
                    progressContainer.style.display = 'block';
                    imgProcessed.style.display = 'block';
                    initInfo.style.display = 'none';
                    videoPreview.play();
                    init = true;
                }
                imgProcessed.src = 'data:image/jpeg;base64,' + event.data;
            }
        }
    }

    videoFile.addEventListener("change", function (event) {
        const file = event.target.files[0];
        if (file) {
            loadVideoInfo.style.display = 'none';
            videoPreview.style.display = 'block';
            videoPreview.src = URL.createObjectURL(file);
            videoPreview.load();

            filename = file.name;
        }
    });

    reuploadBtn.addEventListener('click', function () {
        location.reload();
    });

    ddItems.forEach(item => {
        item.addEventListener('click', function (e) {
            e.preventDefault(); // Previene el salto por el href="#"

            let selectedText = this.textContent.trim();
            if (!selectedText.includes('Sin saltos')) {
                framesSkipToAnalyze = parseInt(selectedText.match(/\d+/)[0]);
                if (selectedText.includes("defecto")) selectedText = selectedText.replace('(Por defecto)', '');
                ddMenuButton.innerHTML = `<i class="fa-solid fa-sliders"></i> Analizar cada: ${selectedText} `;
            } else {
                framesSkipToAnalyze = 0;
                ddMenuButton.innerHTML = `<i class="fa-solid fa-sliders"></i> ${selectedText} `;
            }

            if (framesSkipToAnalyze < 3) {
                labelWarning.style.display = 'block';
            } else {
                labelWarning.style.display = 'none';
            }
        });
    });

    viewResultsBtn.addEventListener('click', function () {
        if (!detecciones.length) {
            showToast("No hay resultados para mostrar", "info");
            return;
        }
        setProcessedVideo();
        modal.style.display = "block";
    });

    closeModalBtn.addEventListener('click', function () {
        modal.style.display = "none";
    });

    document.addEventListener('keyup', function (event) {
        if (event.key === 'Escape' && modal.style.display === 'block') {
            modal.style.display = "none";
        }
    });

    function setProcessedVideo() {
        fetch(`http://localhost:5000/video/${filename}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error("Video no encontrado");
                }
                return response.blob();
            })
            .then(blob => {
                const videoUrl = URL.createObjectURL(blob); // crear URL temporal
                fullVideo.src = videoUrl;
                fullVideo.load();
                //fullVideo.play();
            })
            .catch(error => {
                console.error("Error al cargar el video:", error);
                showToast("Error al cargar el video procesado", "error");
            });
    }
});