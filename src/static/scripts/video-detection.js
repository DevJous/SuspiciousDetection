var filename = '';
var isProcessing = false;

document.addEventListener('DOMContentLoaded', function () {
    const uploadForm = document.getElementById('upload-form');
    const videoFile = document.getElementById('video-file');
    const uploadBtn = document.getElementById('upload-btn');
    const reuploadBtn = document.getElementById('reupload-btn');
    const initInfo = document.getElementById('init-info');
    const progressContainer = document.getElementById('progress-container');
    const videoContainer = document.getElementById('video-container');
    const resultsContainer = document.getElementById('results-container');
    const processedVideo = document.getElementById('processed-video');
    const timestampsList = document.getElementById('timestamps-list');
    const imgProcessed = document.getElementById('img-processed-video');
    const videoPreview = document.getElementById("loaded-video");

    let frameIndex = 1;
    let failedAttempts = 0;
    let seconds_to_wait = 5; // tiempo maximo de espera en segundos cuando no recibe respuesta positiva
    const retryInterval = 100; // tiempo en milisegundos entre cada intento de carga de frame
    const maxFailedAttempts = seconds_to_wait * 1000 / retryInterval; // maximo de intentos fallidos permitidos

    function actualizarFrame() {
        try {
            const filename = `frame_${String(frameIndex).padStart(6, '0')}.jpg`;
            const frameURL = `/frames/${removeFileExtension(this.filename)}/${filename}?` + Date.now();
            const testImg = new Image();

            testImg.onload = () => {
                imgProcessed.src = frameURL;
                frameIndex++;
                failedAttempts = 0; // reiniciar fallos si se carga exitosamente
                setTimeout(actualizarFrame, retryInterval);
            };

            testImg.onerror = () => {
                if (!isProcessing) failedAttempts++;
                if (failedAttempts < maxFailedAttempts) {
                    setTimeout(actualizarFrame, retryInterval);
                } else {
                    console.log(`Detenido: no se encontraron nuevos frames en ${seconds_to_wait} segundos.`);
                    failedAttempts = 0; // reiniciar fallos para el siguiente ciclo
                }
            };

            testImg.src = frameURL;
        } catch (error) {
            // do nothing
        }
    }

    function removeFileExtension(filename) {
        return filename.substring(0, filename.lastIndexOf(".")) || filename;
    }

    videoFile.addEventListener("change", function (event) {
        const file = event.target.files[0];
        if (file) {
            videoPreview.src = URL.createObjectURL(file);
            videoPreview.load();

            videoContainer.style.display = 'flex';
            filename = file.name;
        }
    });

    uploadForm.addEventListener('submit', function (e) {
        e.preventDefault();

        initInfo.style.display = 'none';
        imgProcessed.style.display = 'block';
        process_video();
    });

    reuploadBtn.addEventListener('click', function () {
        location.reload();
    });

    function process_video() {
        if (!videoFile.files[0]) {
            alert('Por favor seleccione un archivo de video');
            return;
        }
        uploadBtn.disabled = true;
        videoFile.disabled = true;
        showToast("Procesando video", "info");

        frameIndex = 1; // Reiniciar el índice del frame
        isProcessing = true;
        actualizarFrame();
        videoPreview.playbackRate = 0.5; // 0.5x (lento) - 1.0x (normal) - 1.5x (rapido) - 2.0x (muy rapido)
        videoPreview.play();

        const formData = new FormData();
        formData.append('video', videoFile.files[0]);

        // Mostrar progreso y desactivar botón
        uploadBtn.disabled = true;
        progressContainer.style.display = 'block';

        // Enviar solicitud AJAX
        fetch('/new-upload', {
            method: 'POST',
            body: formData
        }).then(response => {
            if (!response.ok) {
                throw new Error('Error en el servidor');
            }

            showToast("Video procesado correctamente", "success");
            reuploadBtn.style.display = 'inline-block';
            isProcessing = false;
            return response.json();
        }).then(data => {
            // Procesar respuesta
            progressContainer.style.display = 'none';

            // Mostrar detecciones
            displayDetections(data.detections);
            resultsContainer.style.display = 'block';
        }).catch(error => {
            progressContainer.style.display = 'none';
            reuploadBtn.style.display = 'inline-block';
            showToast(error.message, "error");
            isProcessing = false;
        });
    }

    function displayDetections(detections) {
        // Limpiar lista anterior
        timestampsList.innerHTML = '';

        if (detections.length === 0) {
            timestampsList.innerHTML = '<p class="text-center">No se detectaron comportamientos sospechosos.</p>';
            return;
        }

        // Mostrar cada detección
        detections.forEach((detection, index) => {
            const item = document.createElement('div');
            item.className = 'timestamp-item';
            item.dataset.time = detection.timestamp;

            // Formatear tiempo (segundos a MM:SS)
            const minutes = Math.floor(detection.timestamp / 60);
            const seconds = Math.floor(detection.timestamp % 60);
            const formattedTime = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;

            // Crear HTML para comportamientos
            let behaviorsHTML = '';
            detection.behaviors.forEach(behavior => {
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
                    case 'erratic_movements':
                        label = 'Movimientos erráticos';
                        className = 'erratic-movements';
                        break;
                }

                behaviorsHTML += `<span class="behavior-tag ${className}">${label}</span>`;
            });

            item.innerHTML = `
                        <div><strong>${formattedTime}</strong> - ${behaviorsHTML}</div>
                    `;

            // Agregar evento click para saltar al tiempo del video
            item.addEventListener('click', function () {
                processedVideo.currentTime = detection.timestamp;
                processedVideo.play();
            });

            timestampsList.appendChild(item);
        });
    }
});