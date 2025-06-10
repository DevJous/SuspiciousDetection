//// Poner logica aqui

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const startButton = document.getElementById('startButton');

// Parámetro de duración de captura
const captureDuration = 10; // en segundos
const captureInterval = 1000; // cada 1 segundo (1000 ms)

let captureTimer = null;
let secondsElapsed = 0;

// Accede a la cámara del usuario
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(_ => showToast("No se pudo acceder a la camara", "error"));

function captureAndSendFrame() {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL('image/jpeg');

    fetch('/process_frame', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: dataUrl })
    })
        .then(res => res.json())
        .then(data => {
            console.log('Detección:', data);
            // Puedes mostrar resultados aquí si quieres
        });
}

function startCaptureProcess() {
    secondsElapsed = 0;
    captureTimer = setInterval(() => {
        if (secondsElapsed >= captureDuration) {
            clearInterval(captureTimer);
            alert('Captura finalizada');
        } else {
            captureAndSendFrame();
            secondsElapsed++;
        }
    }, captureInterval);
}

startButton.addEventListener('click', () => {
    if (captureTimer) {
        clearInterval(captureTimer); // en caso de que se haya quedado activo
    }
    startCaptureProcess();
});
