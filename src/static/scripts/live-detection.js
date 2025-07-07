var deteccionesTiempoReal = [];
var framesSkipToAnalyze = 3; // por defecto
var isProcessing = false;
var selDimension = '2D';
var selDeviceType = 'local'
var detecciones = [];

document.addEventListener('DOMContentLoaded', function () {
    const uploadForm = document.getElementById('upload-form');
    const uploadBtn = document.getElementById('upload-btn');
    const reuploadBtn = document.getElementById('reupload-btn');
    const initInfo = document.getElementById('init-info');
    const processedVideo = document.getElementById('full-video');
    const timestampsList = document.getElementById('timestamps-list');
    const imgProcessed = document.getElementById('img-processed-video');
    const skipframes_ddMenuButton = document.getElementById("dropdown1");
    const skipframes_ddItems = document.querySelectorAll('#dd1 .dropdown-item');
    const dimension_ddMenuButton = document.getElementById("dropdown2");
    const dimension_ddItems = document.querySelectorAll('#dd2 .dropdown-item');
    const device_ddMenuButton = document.getElementById("device-dd-btn");
    const device_ddItems = document.querySelectorAll('#device-dd-container .dropdown-item');
    const device_input = document.getElementById("device-url");
    const labelWarning = document.getElementById('label-warning');
    const viewResultsBtn = document.getElementById('view-results');
    const closeModalBtn = document.getElementById('close-btn');
    const fullVideo = document.getElementById('full-video');
    const modal = document.getElementById('myModal');
    const realtime_count = document.getElementById('realtime-results-count');
    const realtime_list = document.getElementById('realtime-list');
    const policeImage = document.getElementById('police-image');
    const irSubContainer = document.getElementById('ir-info-subcontainer');
    const foundResults = document.getElementById('found-results');
    const realtimeSpinner = document.getElementById('realtime-spinner');
    const validate_uri = document.getElementById('validate-uri-info');

    uploadForm.addEventListener('submit', function (e) {
        e.preventDefault();
        if((selDeviceType === 'remote' && validarURLMultimedia(device_input.value)) || selDeviceType === 'local') {
            goToDetection()
            skipframes_ddMenuButton.disabled = true;
            dimension_ddMenuButton.disabled = true;
            uploadBtn.disabled = true;
            irSubContainer.style.display = 'none';
            foundResults.style.display = 'block';
            showToast("Procesando dispositivo de captura", "info");
        } else {
            showToast('Conexión ingresada es erronea o está vacia', 'error')
        }
    })

    device_input.addEventListener('keyup', _ => {
        if(!validarURLMultimedia(device_input.value)) {
            if(device_input.value !== '') {
                validate_uri.style.display = 'block';
            } else {
                validate_uri.style.display = 'none';
            }
        } else {
            validate_uri.style.display = 'none';
        }
    })

    function goToDetection() {
        init = false;
        const source = new EventSource(`/camera_stream_frames/${framesSkipToAnalyze}/${selDimension}/${fixUrlForGet(device_input.value)}`);

        source.onmessage = function (event) {
            if (event.data === "EOF") {
                source.close();
                realtimeSpinner.style.display = 'none';
                showToast("Detección en vivo culminada correctamente", "success");
                fetch('/detecciones')
                    .then(res => res.json())
                    .then(data => {
                        detecciones = data;
                        displayDetections();
                    });
            } else {
                const data = JSON.parse(event.data);
                if (event.data && !init) {
                    reuploadBtn.style.display = 'inline-block';
                    imgProcessed.style.display = 'block';
                    initInfo.style.display = 'none';
                    init = true;
                }
                imgProcessed.src = 'data:image/jpeg;base64,' + data.frame;
                if (data.detections) updateRealTimeDetections(data.detections);
            }
        }
    }

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


    device_ddItems.forEach(item => {
        item.addEventListener('click', function (e) {
            e.preventDefault();

            let selectedText = this.textContent.trim();

            if (selectedText.includes('local')) {
                selDeviceType = 'local'
                device_input.disabled = true;
                device_input.placeholder = 'Se tomará el dispositivo local para el análisis';
            } else {
                selDeviceType = 'remote'
                device_input.disabled = false;
                device_input.placeholder = 'Ingrese la cadena de conexión al dispositivo de transmisión remoto (HTTP - RTSP)';
            }

            device_ddMenuButton.innerText = selectedText;
        })
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
                        label = 'Manos ocultas detrás';
                        className = 'hidden-hands';
                        break;
                    case 'hand_under_clothes':
                        label = 'Mano bajo ropa';
                        className = 'hand-under-clothes';
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
                    realtime_list.innerHTML += `${realtime_list.innerText !== '' ? '<br>' : ''}<label><i class="fa-solid fa-hand-paper"></i>Manos ocultas detrás</label>`;
                    break;
                case 'hand_under_clothes':
                    realtime_list.innerHTML += `${realtime_list.innerText !== '' ? '<br>' : ''}<label><i class="fa-solid fa-camera"></i>Mano oculta bajo ropa</label>`;
                    break;
                default:
                    break;
            }
        }
    }

    function validarURLMultimedia(url) {
        const regex = /^(rtsp|http|https):\/\/[^\s/$.?#].[^\s]*$/i;

        return regex.test(url);
    }

    /// BREVE RESUMEN
    // La peticion y respuesta continua de JS con Flask solo es posible mediante una peticion GET
    // El envio de parametros o enrutamientos van seguidos de un '/', este metodo es una
    // manera de enmascarar la url para evitar que se tome como toda la direccion como varios parametros 
    function fixUrlForGet(url) {
        return url ? url.replaceAll('/', '{slash}') : null
    }
});