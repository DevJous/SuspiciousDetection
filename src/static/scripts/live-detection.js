/*
document.addEventListener('DOMContentLoaded', function() {
    const videoFeed = document.getElementById('videoFeed');
    const toggleCameraBtn = document.getElementById('toggleCamera');
    const statusDiv = document.getElementById('status');
    const alertList = document.getElementById('alertList');

    let cameraActive = false;
    let alertHistory = [];

    // Activar/desactivar cámara
    toggleCameraBtn.addEventListener('click', function() {
        cameraActive = !cameraActive;

        if (cameraActive) {
            videoFeed.src = "{{ url_for('video_feed') }}";
            statusDiv.textContent = "Cámara activada - Detectando comportamientos";
            statusDiv.style.color = "green";
            toggleCameraBtn.textContent = "Desactivar Cámara";

            // Simular detección de alertas (en un sistema real, esto vendría del servidor)
            simulateAlerts();
        } else {
            videoFeed.src = "";
            statusDiv.textContent = "Cámara desactivada";
            statusDiv.style.color = "red";
            toggleCameraBtn.textContent = "Activar Cámara";
        }
    });

    // Simular detección de alertas (para demostración)
    function simulateAlerts() {
        if (!cameraActive) return;

        const behaviors = ["Merodeo", "Pelea", "Intrusión", "Robo", "Vandalismo"];
        const randomBehavior = behaviors[Math.floor(Math.random() * behaviors.length)];
        const timestamp = new Date().toLocaleTimeString();

        const alert = {
            behavior: randomBehavior,
            time: timestamp,
            location: "Zona " + (Math.floor(Math.random() * 5) + 1)
        };

        alertHistory.unshift(alert);
        if (alertHistory.length > 5) {
            alertHistory.pop();
        }

        updateAlertList();

        // Volver a llamar después de un intervalo aleatorio
        setTimeout(simulateAlerts, Math.random() * 10000 + 5000);
    }

    function updateAlertList() {
        alertList.innerHTML = '';
        alertHistory.forEach(alert => {
            const li = document.createElement('li');
            li.innerHTML = `<strong>${alert.behavior}</strong> - ${alert.location} (${alert.time})`;
            alertList.appendChild(li);
        });
    }

    // Solicitar permiso para la cámara solo cuando el usuario lo active
    function requestCameraPermission() {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(function(stream) {
                // Permiso concedido, pero no hacemos nada hasta que el usuario active la cámara
                console.log("Permiso de cámara concedido");
            })
            .catch(function(err) {
                console.error("Error al acceder a la cámara: ", err);
                statusDiv.textContent = "Error al acceder a la cámara";
                statusDiv.style.color = "red";
            });
    }

    // Solicitar permiso cuando se haga clic en el botón por primera vez
    toggleCameraBtn.addEventListener('click', requestCameraPermission, { once: true });
});*/

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const processed = document.getElementById('processedFeed');

// Captura la cámara
navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
    video.srcObject = stream;

    // Empieza a capturar imágenes cada 1 segundo
    setInterval(() => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);

        canvas.toBlob((blob) => {
            const formData = new FormData();
            formData.append('frame', blob, 'frame.jpg');

            fetch('/video_feed', {
                method: 'POST',
                body: formData
            }).then(response => response.blob())
                .then(blob => {
                const url = URL.createObjectURL(blob);
                processed.src = url;

                // Limpieza
                setTimeout(() => URL.revokeObjectURL(url), 100);
            });
        }, 'image/jpeg');
    }, 100); // cada 1 segundo
})
    .catch((err) => {
    console.error('No se pudo acceder a la cámara:', err);
});