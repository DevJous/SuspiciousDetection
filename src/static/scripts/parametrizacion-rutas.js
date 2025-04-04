const main_content_view = document.getElementById('main_container_routes');
const folderPath = document.getElementById('folderInput');
const container_button_save = document.getElementById('container_button_save');
const buttonSave = document.getElementById('button_save');
const closeModalButton = document.getElementById('save');
const modalCreateFolder = document.getElementById('modal_create_folder');
const message = document.getElementById('message_modal');
const inputCreateFolder = document.getElementById('inputFolderName');
const deleteButtons = document.getElementById('deletePath');
const modalOptions = document.getElementById('modal_delete_path');
const deleteButtonOption = document.getElementById('btnDeleteOption');

const iconSaveNewFrame = document.getElementById('iconSaveNewFrame');
const iconEditFrame = document.getElementById('icon_edit_button');
const iconCancelFrame = document.getElementById('icon_cancel_option');

const txtNumberFPS = document.getElementById('txtNumberFPS');


var main_path = '';
let opcionActual = '';

document.addEventListener("DOMContentLoaded", async function () {
    const token = localStorage.getItem('token');
    setTimeout(() => {
        if (!token) {
            window.location.href = '/login';
            return;
        }
    }, 100);
    await HasPath();
    await cargarRutas();
    let frame = localStorage.getItem('frames');
    txtNumberFPS.disabled = true;
    txtNumberFPS.placeholder = localStorage.getItem('frames');

});

async function HasPath(){
    const form = new FormData()
    form.append('id', localStorage.getItem('id'))

    try{
        const response = await fetch('/validate_has_path',{
            method: 'POST',
            body: form
        });
        const data = await response.json();
        if(data.ruta){
            folderPath.placeholder = '';
            folderPath.disabled = true;
            buttonSave.style.display = "none";
            buttonSave.disabled = true
            container_button_save.style.display = "none"; 
            folderPath.value = data.ruta;
            main_path = data.ruta;
        }else {
            folderPath.placeholder = 'Pega aqui la ruta que tendra tu proyecto completo'
        }

    }catch(error){
        console.log(error)
    }
}

async function saveMainPath(){
    const form = new FormData();
    let valid = true;
    const regexRutaWindows = /^[a-zA-Z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*$/;
    const rutaIngresada = folderPath.value;
    console.log(rutaIngresada);
    form.append('id', localStorage.getItem('id'));

    if(!regexRutaWindows.test(rutaIngresada)){
        valid = false;
    } else if(rutaIngresada == ''){
        valid = false;
    } else {
        form.append('path',rutaIngresada);
    }

    if (!valid) {
        console.error('El formulario de ruta contiene errores y no se enviará.');
        return;
    }

    try{
        const request = await fetch('/parametrizador-ruta-principal',{
            method: 'POST',
            body: form
        });

        const response = await request.json();
        if(response.created){
            location.reload();
        }

    }catch(error){
        console.log(error)
    }
}

async function cargarRutas() {
    try {
        let data = new FormData();
        data.append('id', localStorage.getItem('id'));

        const response = await fetch('/all_paths', {
            method: 'POST',
            body: data
        });
        const datos = await response.json();
        console.log(datos)
        const tablaCuerpo = document.getElementById('tabla-cuerpo');
        tablaCuerpo.style.position = 'relative';

        if(datos.length != 0){
            datos.forEach(fila => {
                const filaTabla = document.createElement('tr');
                filaTabla.innerHTML = `
                    <td>${fila.id}</td>
                    <td>${fila.nombre}</td>
                    <td>${fila.fechaCreacion}</td>
                    <td>
                        <button id="deletePath" class="delete-path-btn" style="background-color:red; cursor:pointer; border-radius:4px;" onclick="deletePath(this)">
                            <i class="bi bi-trash3-fill" style="color:#ffffff; margin:0 auto;"></i>
                        </button>
                    </td>
                `;
                tablaCuerpo.appendChild(filaTabla);
            });
        }else{
            const filaTabla = document.createElement('tr');
            filaTabla.innerHTML = `<td style="text-align:center; position:absolute; width:100%;">Actualmente no tienes rutas registradas</td>`
            tablaCuerpo.appendChild(filaTabla);
        }
        // Muestra la tabla y oculta el mensaje de carga
    } catch (error) {
        console.error('Error al cargar los datos:', error);
    }
}

async function saveNewFolder(){
    const data = new FormData();
    data.append('main_path',this.main_path);

    const request = await fetch('/getIdMainPath',{
        method: 'POST',
        body : data
    });

    //OBTENER ID DE RUTA PRINCIPAL
    let response = await request.json()


    const savefolder = new FormData();
    savefolder.append('nameFolder',this.main_path+'\\'+inputCreateFolder.value);
    savefolder.append('id_main_folder',response.id_path);

    const request_folder = await fetch('/save_new_folder',{
        method: 'POST',
        body : savefolder
    });
    let response_folder = await request_folder.json();
    let message = response_folder.message;


    if(response_folder.created){
        generateMessageSuccesfull(message);
        let idUser = localStorage.getItem('id');
        await getPaths(idUser);
        location.reload();

    }else{
        generateMessageError(message);
    }
    closeModal();
}

function generateMessageSuccesfull(message){
    const messageDiv = document.createElement('div');
    messageDiv.textContent = message;
    messageDiv.style.position = 'fixed';
    messageDiv.style.bottom = '4%';
    messageDiv.style.right = '4%';
    messageDiv.style.backgroundColor = 'rgba(103, 220, 17 ,0.8)';
    messageDiv.style.padding = '10px 20px';
    messageDiv.style.borderRadius = '8px';
    messageDiv.style.boxShadow = '0 2px 6px rgba(0, 0, 0, 0.2)';
    messageDiv.style.fontSize = '16px';
    messageDiv.style.zIndex = '1000';
    messageDiv.style.transition = 'opacity 0.5s';

    document.body.appendChild(messageDiv);

    setTimeout(() => {
        messageDiv.remove();
    }, 5000);
}

function generateMessageError(message){
    const messageDiv = document.createElement('div');
    messageDiv.textContent = message;
    messageDiv.style.position = 'fixed';
    messageDiv.style.bottom = '4%';
    messageDiv.style.right = '4%';
    messageDiv.style.backgroundColor = '#da0b0b';
    messageDiv.style.padding = '10px 20px';
    messageDiv.style.borderRadius = '8px';
    messageDiv.style.boxShadow = '0 2px 6px rgba(0, 0, 0, 0.2)';
    messageDiv.style.fontSize = '16px';
    messageDiv.style.zIndex = '1000';
    messageDiv.style.transition = 'opacity 0.5s';

    document.body.appendChild(messageDiv);

    setTimeout(() => {
        messageDiv.remove();
    }, 5000);
}

function deletePath(button){
    let rows = button.closest('tr');
    let rowData = Array.from(rows.querySelectorAll('td')).map(td => td.textContent);
    openModalOptions(rowData[1]);
}


//modals actions
function closeModal(){
    modalCreateFolder.style.display = "none";
}

function openModalCreatePath(){
    modalCreateFolder.style.display = "block";
}

function closeModalOptions(){
    modalOptions.style.display = "none";
}

async function openModalOptions(rowData){
    modalOptions.style.display = "block";

    deleteButtonOption.addEventListener("click", async() =>{
        let request = new FormData();
        request.append('path',rowData);

        const response = await fetch('/delete_folder', {
            method: 'POST',
            body: request
        });

        var data = await response.json();
        generateMessageSuccesfull(data.message);
        closeModalOptions();
        window.location.href = window.location.href;
    });

}

    function enabledSelectFrame(){
        txtNumberFPS.disabled = false;
        iconSaveNewFrame.style.display = 'block';
        iconEditFrame.style.display = 'none';
        iconCancelFrame.style.display = 'block';
    }

    function cancelSelectFrame(){
        txtNumberFPS.disabled = true;
        iconSaveNewFrame.style.display = 'none';
        iconEditFrame.style.display = 'block';
        iconCancelFrame.style.display = 'none';
    }


    async function getPaths(idUser){
        try{
            let form = new FormData();
            console.log(idUser)
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

    async function saveNewFrame() {
         const form = new FormData();
         form.append('frame_value',txtNumberFPS.value)
         form.append('id_user', localStorage.getItem('id'));

         if(txtNumberFPS.value >= 1 && txtNumberFPS.value <= 24)
         {
             try {
                 const response = await fetch('/saveNewFrame', {
                    method: 'POST',
                    body: form
                 });

                 if (!response.ok) {
                    throw new Error(`Error saving frame: ${response.statusText}`);
                 }

                 const data = await response.json();
                 localStorage.setItem('frames', txtNumberFPS.value);
                 generateMessageSuccesfull(data.message);
                 txtNumberFPS.disabled = true;
                 iconSaveNewFrame.style.display = 'none';
                 iconEditFrame.style.display = 'block';
                 iconCancelFrame.style.display = 'none';

             } catch (error) {
                 console.error('Error saving frame:', error);
             }
         }else{
            const messageDiv = document.createElement('div');
            messageDiv.textContent = 'No puedes ingresar un valor fuera del rango 1 - 24';
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


    async function getPaths(idUser){
        try{
            let form = new FormData();
            console.log(idUser)
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
