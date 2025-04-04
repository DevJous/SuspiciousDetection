const txtPath = document.getElementById('txtPath');
const txtFPS = document.getElementById('txtFPS');
const modal_first_session = document.getElementById('modal_first_session');

var stateTutorial = '';

document.addEventListener("DOMContentLoaded", async function () {
    const token = localStorage.getItem('token');
    await validateHasTutorial(token);

    setTimeout(() => {
        if (!token) {
            window.location.href = '/login';
            return;
        }

        const nameCache = localStorage.getItem('user');
        if (nameCache) {
            const name_m = document.getElementById('userTutorial');
            if (name_m) {
                name_m.textContent = nameCache;
            } else {
                console.log('No se pudo encontrar el elemento con id="name-user"');
            }
        } else {
            console.log('No se encontró el valor "user" en localStorage');
        }



        if(stateTutorial === '0'){
            modal_first_session.style.display = 'block';
        }
    }, 100);
});

async function saveFirstTutorial(){
    const form = new FormData();
    const regexRutaWindows = /^[a-zA-Z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*$/;
    const rutaIngresada = txtPath.value;
    const valid = true;
    console.log(rutaIngresada);

    if(rutaIngresada != null && rutaIngresada != "" && txtFPS.value != null && txtFPS.value != ""){
        if(rutaIngresada != null || rutaIngresada != "" || txtFPS.value != null || txtFPS.value != "" ){
            form.append('id_user',localStorage.getItem('id'));
            form.append('main_path',rutaIngresada);
            if(txtFPS.value != '' || txtFPS.value != null){
                form.append('fps_value',txtFPS.value);
            }

            if(!regexRutaWindows.test(rutaIngresada)){
                valid = false;
            } else if(rutaIngresada == ''){
                valid = false;
            } else {
                form.append('path',rutaIngresada);
            }

            if (!valid) {
                const messageDiv = document.createElement('div');
                messageDiv.textContent = 'El formulario de ruta contiene errores y no se enviará';
                messageDiv.style.position = 'fixed';
                messageDiv.style.bottom = '20px';
                messageDiv.style.right = '20px';
                messageDiv.style.backgroundColor = 'rgb(239, 79, 61)';
                messageDiv.style.color = '#fff';
                messageDiv.style.padding = '10px 20px';
                messageDiv.style.borderRadius = '8px';
                messageDiv.style.boxShadow = '0 2px 6px rgba(0, 0, 0, 0.2)';
                messageDiv.style.fontSize = '16px';
                messageDiv.style.zIndex = '1000';
                messageDiv.style.transition = 'opacity 0.5s';

                document.body.appendChild(messageDiv);

                setTimeout(() => {
                    messageDiv.style.opacity = '0';
                    setTimeout(() => {
                        messageDiv.remove();
                    }, 500);
                }, 3000);
                return;
            }

            const request = await fetch('/saveFirstTutorialInfo',{
                 method: 'POST',
                 body: form
            });

            const response = await request.json();

            const messageDiv = document.createElement('div');
            messageDiv.textContent = 'Configuración inicial guardada correctamente';
            messageDiv.style.position = 'fixed';
            messageDiv.style.bottom = '20px';
            messageDiv.style.right = '20px';
            messageDiv.style.backgroundColor = 'rgb(91, 239, 61)';
            messageDiv.style.color = '#fff';
            messageDiv.style.padding = '10px 20px';
            messageDiv.style.borderRadius = '8px';
            messageDiv.style.boxShadow = '0 2px 6px rgba(0, 0, 0, 0.2)';
            messageDiv.style.fontSize = '16px';
            messageDiv.style.zIndex = '1000';
            messageDiv.style.transition = 'opacity 0.5s';

            document.body.appendChild(messageDiv);

            setTimeout(() => {
                messageDiv.style.opacity = '0';
                setTimeout(() => {
                    messageDiv.remove();
                }, 500);
            }, 3000);

            modal_first_session.style.display = 'none';
            localStorage.setItem('frames',txtFPS.value);
        }else{
            const messageDiv = document.createElement('div');
            messageDiv.textContent = 'Verifica que ambos campos contengan datos';
            messageDiv.style.position = 'fixed';
            messageDiv.style.bottom = '20px';
            messageDiv.style.right = '20px';
            messageDiv.style.backgroundColor = 'rgb(241, 2, 2)';
            messageDiv.style.color = '#fff';
            messageDiv.style.padding = '10px 20px';
            messageDiv.style.borderRadius = '8px';
            messageDiv.style.boxShadow = '0 2px 6px rgba(0, 0, 0, 0.2)';
            messageDiv.style.fontSize = '16px';
            messageDiv.style.zIndex = '1000';
            messageDiv.style.transition = 'opacity 0.5s';

            document.body.appendChild(messageDiv);

            setTimeout(() => {
                messageDiv.style.opacity = '0';
                setTimeout(() => {
                    messageDiv.remove();
                }, 500);
            }, 3000);
        }
    }else{
        const messageDiv = document.createElement('div');
        messageDiv.textContent = 'Verifica que ambos campos contengan datos';
        messageDiv.style.position = 'fixed';
        messageDiv.style.bottom = '20px';
        messageDiv.style.right = '20px';
        messageDiv.style.backgroundColor = 'rgb(241, 2, 2)';
        messageDiv.style.color = '#fff';
        messageDiv.style.padding = '10px 20px';
        messageDiv.style.borderRadius = '8px';
        messageDiv.style.boxShadow = '0 2px 6px rgba(0, 0, 0, 0.2)';
        messageDiv.style.fontSize = '16px';
        messageDiv.style.zIndex = '1000';
        messageDiv.style.transition = 'opacity 0.5s';

        document.body.appendChild(messageDiv);

        setTimeout(() => {
            messageDiv.style.opacity = '0';
            setTimeout(() => {
                messageDiv.remove();
            }, 500);
        }, 3000);
    }
    
}

async function validateHasTutorial(token){

    const form = new FormData();
    form.append('id', localStorage.getItem('id'));

    const request = await fetch('/getTutorialState',{
         method: 'POST',
         headers: {
             'Authorization': `Bearer ${token}`,
         },
         body: form
    });

    if (request.status === 401) {
        alert('Tu sesión ha expirado o el token no es válido. Por favor, inicia sesión nuevamente.');
        localStorage.removeItem('token');
        window.location.href = '/login';
        return;
    }

    const response = await request.json();
    stateTutorial = response.state_tutorial;
}