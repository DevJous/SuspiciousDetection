const inputMail = document.getElementById('mail');
const inputPassword = document.getElementById('password');
const emailMessageError = document.getElementById('emailMessage');
const passMessageError = document.getElementById('passMessage');
const loader = document.getElementById('loader_container');
const iconViewPassword = document.getElementById('iconPassword');

iconViewPassword.addEventListener('click', ()=>{
    const isPassword = inputPassword.type === 'password';
    inputPassword.type = isPassword ? 'text' : 'password';
});

window.onload = function() {
    const message = sessionStorage.getItem('successMessage');
    if (message) {
        messageInformation(message); // Muestra el mensaje
        sessionStorage.removeItem('successMessage'); // Borra el mensaje tras mostrarlo

        // Oculta el mensaje tras unos segundos
        setTimeout(() => {
            messageInformation('');
        }, 4000);
    }
};

    async function getLogin() {
        const form = new FormData();
        let ruta;
        let valid = true;
        loader.style.display = 'block';


        const email = inputMail.value;
        const response = await fetch('/check_email', {
            method: 'POST',
            body: JSON.stringify({ email }),
            headers: {
                'Content-Type': 'application/json'
            }
        })

        const data = await response.json();
        // Validar correo
        if (email === '') {
            loader.style.display = 'none';
            emailMessageError.textContent = 'Debe ingresar su correo';
            valid = false;
        } else if (!data.exists) {
            loader.style.display = 'none';
            valid = false;
            emailMessageError.textContent = 'El correo que ingreso no existe';
        } else {
            loader.style.display = 'none';
            form.append('mail', email);
            emailMessageError.textContent = ''; // Limpiar mensaje de error
        }

        const pass = inputPassword.value;
        if (pass === '') {
            loader.style.display = 'none';
            valid = false;
            passMessageError.textContent = 'Debe ingresar su contraseña';
        } else {
            loader.style.display = 'none';
            form.append('pass', pass);
            passMessageError.textContent = '';
        }

        if (!valid) {
            loader.style.display = 'none';
            console.error('El formulario contiene errores y no se enviará.');
            return;
        }

        try {
            const response = await fetch('/validateLogin', {
                method: 'POST',
                body: form
            });
            const data = await response.json();
            console.log(data)
            if (data.authenticated) {
                this.ruta = data.redirect_url;
                localStorage.setItem('token',data.token)
                localStorage.setItem('user', data.user)
                localStorage.setItem('id', data.id);
                localStorage.setItem('rol', data.idRol)
                let idUser = data.id;
                await getPaths(idUser);
                await getFrames(data.id);
                loader.style.display = 'none';

                window.location.href = this.ruta;

            } else {
                if(data.stateuser === '1'){
                    loader.style.display = 'none';

                    document.getElementById('message_content').innerHTML =
                    `
                        <div style="position:absolute; padding:10px 30px; top:5%; left:50%; transform: translateX(-50%); background-color:rgb(214, 5, 5 );">
                            <p style="text-align:center; color:#ffffff;">${data.message}</p>
                        </div>
                    `
                    setTimeout(() => {
                        document.getElementById('message_content').innerHTML = '';
                    }, 4000);
                } else if (data.stateuser === '0'){
                    loader.style.display = 'none';

                    document.getElementById('message_content').innerHTML =
                    `
                        <div style="position:absolute; padding:10px 30px; top:5%; left:50%; transform: translateX(-50%); background-color:rgb(214, 5, 5 );">
                            <p style="text-align:center; color:#ffffff;">Al parecer tu cuenta esta deshabilitada. Comunicate con un administrador.</p>
                        </div>
                    `
                    setTimeout(() => {
                        document.getElementById('message_content').innerHTML = '';
                    }, 4000);
                }
            }
        } catch (error) {
        console.error('Error:', error);
        loader.style.display = 'none';
        document.getElementById('message_content').innerHTML = `
            <div style="position:absolute; padding:10px 30px; top:5%; left:50%; transform: translateX(-50%); background-color:rgb(214, 5, 5 );">
                <p style="text-align:center; color:#ffffff;">Ocurrió un error al iniciar sesión. Inténtalo de nuevo.</p>
            </div>
        `;
        setTimeout(() => {
            document.getElementById('message_content').innerHTML = '';
        }, 4000);
    }
    }

    async function getPaths(idUser){
        try{
            let form = new FormData();
            form.append('id',idUser);

            let request = await fetch('/all_paths',{
                method: 'POST',
                body: form
            });

            let response = await request.json();
            localStorage.setItem('paths', JSON.stringify(response));
        }catch(error){
            console.log(error);
        }
    }

    async function getFrames(idUser){
        try{
            let form = new FormData();
            form.append('id',idUser);

            let request = await fetch('/get_frames',{
                method: 'POST',
                body: form
            });

            let response = await request.json();
            localStorage.setItem('frames', JSON.stringify(response.response));
        }catch(error){
            console.log(error);
        }
    }

    document.getElementById('login-form').addEventListener('keydown', function(event) {
        if (event.key === 'Enter') {
            event.preventDefault();
            document.getElementById('login-button').click(); // Simular el clic en el botón
        }
    });

    function messageInformation(message) {
        const messageDiv = document.createElement('div');
        messageDiv.textContent = message;
        messageDiv.style.position = 'fixed';
        messageDiv.style.bottom = '20px';
        messageDiv.style.right = '20px';
        messageDiv.style.backgroundColor = 'rgb(102, 199, 11)';
        messageDiv.style.color = '#fff';
        messageDiv.style.padding = '10px 20px';
        messageDiv.style.borderRadius = '8px';
        messageDiv.style.boxShadow = '0 2px 6px rgba(250, 250, 250, 0.2)';
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