var deteccionesTiempoReal = [];
var framesSkipToAnalyze = 3; // por defecto
var isProcessing = false;
var selDimension = '2D';
var detecciones = [];
var filename = '';

document.addEventListener('DOMContentLoaded', function () {
    const uploadForm = document.getElementById('upload-form');
    const videoFile = document.getElementById('video-file');
    const uploadBtn = document.getElementById('upload-btn');
    const reuploadBtn = document.getElementById('reupload-btn');
    const initInfo = document.getElementById('init-info');
    const progressContainer = document.getElementById('progress-container');
    const processedVideo = document.getElementById('full-video');
    const timestampsList = document.getElementById('timestamps-list');
    const imgProcessed = document.getElementById('img-processed-video');
    const videoPreview = document.getElementById("loaded-video");
    const skipframes_ddMenuButton = document.getElementById("dropdown1");
    const skipframes_ddItems = document.querySelectorAll('#dd1 .dropdown-item');
    const dimension_ddMenuButton = document.getElementById("dropdown2");
    const dimension_ddItems = document.querySelectorAll('#dd2 .dropdown-item');
    const labelWarning = document.getElementById('label-warning');
    const loadVideoInfo = document.getElementById('load-video-info');
    const viewResultsBtn = document.getElementById('view-results');
    const closeModalBtn = document.getElementById('close-btn');
    const fullVideo = document.getElementById('full-video');
    const progressBar = document.getElementsByClassName('progress')[0];
    const modal = document.getElementById('myModal');
    const realtime_count = document.getElementById('realtime-results-count');
    const realtime_list = document.getElementById('realtime-list');
    const policeImage = document.getElementById('police-image');
    const irSubContainer = document.getElementById('ir-info-subcontainer');
    const foundResults = document.getElementById('found-results');
    const realtimeSpinner = document.getElementById('realtime-spinner');

    uploadForm.addEventListener('submit', function (e) {
        e.preventDefault();
        if (!videoFile.files[0]) {
            showToast("Primero seleccione un archivo de video para continuar", "info");
            return;
        }

        const formData = new FormData();
        formData.append('video', videoFile.files[0]);

        fetch('/save-video', {
            method: 'POST',
            body: formData
        })
            .then(async res => {
                const data = await res.json();
                if (!res.ok) {
                    throw data;
                }
                goToDetection(data.filename)
                skipframes_ddMenuButton.disabled = true;
                dimension_ddMenuButton.disabled = true;
                uploadBtn.disabled = true;
                videoFile.disabled = true;
                irSubContainer.style.display = 'none';
                foundResults.style.display = 'block';
                showToast("Procesando video", "info");
            })
            .catch(err => {
                if (err.message && err.message.includes('existe')) {
                    goToDetection(err.filename);
                    skipframes_ddMenuButton.disabled = true;
                    dimension_ddMenuButton.disabled = true;
                    uploadBtn.disabled = true;
                    videoFile.disabled = true;
                    irSubContainer.style.display = 'none';
                    foundResults.style.display = 'block';
                    showToast("Procesando video", "info");
                } else {
                    showToast("Error al procesar el video", "error");
                }
            });
    })

    function goToDetection(filename) {
        init = false;
        const source = new EventSource(`/stream_frames/${filename}/${framesSkipToAnalyze}/${selDimension}`);

        source.onmessage = function (event) {
            if (event.data === "EOF") {
                source.close();
                realtimeSpinner.style.display = 'none';
                showToast("Video procesado correctamente", "success");
                fetch('/detecciones')
                    .then(res => res.json())
                    .then(data => {
                        //progressContainer.style.display = 'none';
                        detecciones = data;
                        displayDetections();
                    });
            } else {
                const data = JSON.parse(event.data);
                if (event.data && !init) {
                    reuploadBtn.style.display = 'inline-block';
                    progressContainer.style.display = 'block';
                    imgProcessed.style.display = 'block';
                    initInfo.style.display = 'none';
                    videoPreview.play();
                    init = true;
                }
                updateProgressBar(data.progress);
                imgProcessed.src = 'data:image/jpeg;base64,' + data.frame;
                if (data.detections) updateRealTimeDetections(data.detections);
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

    skipframes_ddItems.forEach(item => {
        item.addEventListener('click', function (e) {
            e.preventDefault();

            let selectedText = this.textContent.trim();
            if (!selectedText.includes('Sin saltos')) {
                framesSkipToAnalyze = parseInt(selectedText.match(/\d+/)[0]);
                if (selectedText.includes("defecto")) selectedText = selectedText.replace(' (Por defecto)', '');
                skipframes_ddMenuButton.innerHTML = `<i class="fa-solid fa-sliders"></i> Analizar cada: ${selectedText}&nbsp;`;
            } else {
                framesSkipToAnalyze = 0;
                skipframes_ddMenuButton.innerHTML = `<i class="fa-solid fa-sliders"></i> ${selectedText}&nbsp;`;
            }

            if (framesSkipToAnalyze < 3) {
                labelWarning.style.display = 'block';
            } else {
                labelWarning.style.display = 'none';
            }

        });
    });

    dimension_ddItems.forEach(item => {
        item.addEventListener('click', function (e) {
            e.preventDefault();

            let selectedText = this.textContent.trim();
            selDimension = selectedText;
            dimension_ddMenuButton.innerHTML = `<i class="fa-solid fa-male"></i> Estimación postural: ${selectedText}&nbsp;`;
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
        closeModal()
    });

    document.addEventListener('keyup', function (event) {
        if (event.key === 'Escape' && modal.style.display === 'block') {
            closeModal()
        }
    });

    function closeModal() {
        modal.style.display = "none";
    }

    function setProcessedVideo() {
        fetch(`/processed-video/${filename.replace('.mp4', '.webm')}`)
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

    function updateProgressBar(percentage) {
        progressBar.innerHTML = `<div class="progress-bar progress-bar-striped ${percentage >= 100 ? 'bg-success' : ''}" role="progressbar"
            style="width: ${percentage}%;" aria-valuenow="${percentage}" aria-valuemin="0" aria-valuemax="100">${percentage}%</div>`;
    }

    function displayDetections() {
        // Limpiar lista anterior
        timestampsList.innerHTML = '';

        if (detecciones.length === 0) {
            timestampsList.innerHTML = '<p class="text-center">No se detectaron comportamientos sospechosos.</p>';
            return;
        }

        detecciones.forEach((deteccion, index) => {
            const item = document.createElement('div');
            item.className = 'timestamp-item';
            item.dataset.time = deteccion.timestamp;

            const minutes = Math.floor(deteccion.timestamp / 60);
            const seconds = Math.floor(deteccion.timestamp % 60);
            const formattedTime = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;

            let behaviorsHTML = '';
            deteccion.behaviors.forEach(behavior => {
                let label = '';
                let className = '';

                switch (behavior) {
                    case 'excessive_gaze':
                        label = 'Mirada excesiva';
                        className = 'excessive-gaze';
                        break;
                    case 'hidden_hands':
                        label = 'Manos ocultas';
                        className = 'hidden-hands';
                        break;
                    case 'camera_approach':
                        label = 'Acercamiento indebido';
                        className = 'camera-approach';
                        break;
                }

                behaviorsHTML += `<span class="behavior-tag ${className}">${label}</span>`;
            });

            item.innerHTML = `
                        <div><strong>${formattedTime}</strong> - ${behaviorsHTML}</div>
                    `;

            // Agregar evento click para saltar al tiempo del video
            item.addEventListener('click', function () {
                processedVideo.currentTime = deteccion.timestamp;
                //processedVideo.play();
            });

            timestampsList.appendChild(item);
        });
    }

    function updateRealTimeDetections(detections) {
        if (deteccionesTiempoReal.length >= 3) return;
        deteccionesTiempoReal = [...new Set([...deteccionesTiempoReal, ...detections])]

        if (!policeImage.src.includes('angry')) policeImage.src = '/static/assets/angry_police.jpg';
        if (deteccionesTiempoReal.length > parseInt(realtime_count.innerText)) {
            realtime_count.innerText = deteccionesTiempoReal.length;
            switch (deteccionesTiempoReal[deteccionesTiempoReal.length - 1]) {
                case 'excessive_gaze':
                    realtime_list.innerHTML += `${realtime_list.innerText !== '' ? '<br>' : ''}<label><i class="fa-solid fa-eye"></i>Mirada excesiva</label>`;
                    break;
                case 'hidden_hands':
                    realtime_list.innerHTML += `${realtime_list.innerText !== '' ? '<br>' : ''}<label><i class="fa-solid fa-hand-paper"></i>Manos ocultas</label>`;
                    break;
                case 'camera_approach':
                    realtime_list.innerHTML += `${realtime_list.innerText !== '' ? '<br>' : ''}<label><i class="fa-solid fa-camera"></i>Acercamiento indebido</label>`;
                    break;
                default:
                    break;
            }
        }
    }
});