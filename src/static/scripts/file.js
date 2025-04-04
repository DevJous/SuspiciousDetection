const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadForm = document.getElementById('upload-form');
const previewImg = document.getElementById('preview-img');
const button = document.getElementById('button');
const modalImgPreview = document.getElementById('modal-imgPreview');
const modalImg = document.getElementById('modal-img');
const selectOptions = document.getElementById('selectOptions');
const optionVerticalImage = document.getElementById('select-option-vertical');
const optionHorizontalImage = document.getElementById('select-option-horizontal');
const optionCuadradaImage = document.getElementById('select-option-cuadrada');
const modalEdit = document.getElementById('imageModal');
const containerMiniCards = document.getElementById('content_mini_cards');
const contentOptions = document.getElementById('content-options');
const val_width = document.getElementById('val_width');
const val_height = document.getElementById('val_height');
const val_width_modal = document.getElementById('val_width_modal');
const val_height_modal = document.getElementById('val_height_modal');
const modal_error = document.getElementById('main_content_modal_error');
const buttonSaveImage = document.getElementById('saveButton');


var selectedOption = '';
var selectedOptionPath = '';
var selectedOptionResize = '';
const h = document.getElementById('h');
const w = document.getElementById('w');
let cachedFile = null; 


// Almacena los puntos
let points = [];
var divToPoints = [
                        {'id':'0', 'name':'Nariz', 'divName': document.getElementById('1')},
                        {'id':'11','name':'Hombro Izquierdo' , 'divName': document.getElementById('2')},
                        {'id':'12','name':'Hombro Derecho' , 'divName': document.getElementById('3')},
                        {'id':'13','name':'Codo Izquierdo' , 'divName': document.getElementById('4')},
                        {'id':'14','name':'Codo Derecho' , 'divName': document.getElementById('5')},
                        {'id':'15','name':'Muñeca Izquierda' , 'divName': document.getElementById('6')},
                        {'id':'16','name':'Muñeca Derecha', 'divName': document.getElementById('7')},
                        {'id':'23','name':'Cadera Izquierda', 'divName': document.getElementById('8')},
                        {'id':'24','name':'Cadera Derecha', 'divName': document.getElementById('9')},
                        {'id':'25','name':'Rodilla Izquierda', 'divName': document.getElementById('10')},
                        {'id':'26','name':'Rodilla Derecha', 'divName': document.getElementById('11')},
                        {'id':'27','name':'Tobillo Izquierdo', 'divName': document.getElementById('12')},
                        {'id':'28','name':'Tobillo Derecho', 'divName': document.getElementById('13')},
                        {'id':'33','name':'Centro Pecho', 'divName': document.getElementById('14')}
                       ]
const fileModalImageName = '';
var width_resize;
var height_resize;

    document.addEventListener('DOMContentLoaded', async () => {
        let width, height;
        const token = localStorage.getItem('token');
        setTimeout(() => {
            if (!token) {
                window.location.href = '/login';
                return;
            }
        }, 100);
        await cargarRutas();

        if (selectOptions.options.length > 0) {
            selectOptions.selectedIndex = 0;
            selectedOptionPath = selectOptions.options[0].textContent;
                    console.log('OP1:',selectedOptionPath)

        }
        selectOptions.addEventListener('change', (event) => {
            selectedOptionPath = event.target.selectedOptions[0].textContent;
        });

    });

    optionVerticalImage.addEventListener('change', (event) => {
        selectedOption = event.target.value;

        if (selectedOption === 'option1') {
            width_resize = 175;
            height_resize = 260;
            val_width.textContent = '175'
            val_height.textContent = '260'

        } else if (selectedOption === 'option2') {
            width_resize = 225;
            height_resize = 334;
            val_width.textContent = '225'
            val_height.textContent = '334'

        } else if (selectedOption === 'option3') {
            width_resize = 300;
            height_resize = 445;
            val_width.textContent = '300'
            val_height.textContent = '445'

        }

    });

    optionHorizontalImage.addEventListener('change', (event) => {
        selectedOption = event.target.value;

        if (selectedOption === 'option1') {
            width_resize = 250;
            height_resize = 167;
            val_width.textContent = '250'
            val_height.textContent = '167'

        } else if (selectedOption === 'option2') {
            width_resize = 300;
            height_resize = 200;
            val_width.textContent = '300'
            val_height.textContent = '200'

        } else if (selectedOption === 'option3') {
            width_resize = 350;
            height_resize = 233;
            val_width.textContent = '350'
            val_height.textContent = '233'

        }

    });

    optionCuadradaImage.addEventListener('change', (event) => {
        selectedOption = event.target.value;

        if (selectedOption === 'option1') {
            width_resize = 250;
            height_resize = 250;
            val_width.textContent = '250'
            val_height.textContent = '250'

        } else if (selectedOption === 'option2') {
            width_resize = 300;
            height_resize = 300;
            val_width.textContent = '300'
            val_height.textContent = '300'

        } else if (selectedOption === 'option3') {
            width_resize = 350;
            height_resize = 350;
            val_width.textContent = '350'
            val_height.textContent = '350'

        }
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    // Quita el estilo al salir
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    // Manejador para cuando se suelta el archivo
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');

        // Obtiene el archivo y lo asigna al input
        const files = e.dataTransfer.files[0];
        if (files && files.type.startsWith('image/')) {
            fileInput.files = e.dataTransfer.files; // Asignar al input
            cachedFile = files;
            showPreview(files); // Mostrar la imagen
            toggleButtonState(true);
        }
    });

    // Permitir hacer clic en la zona de drop
    dropZone.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });
    // Cuando seleccionas un archivo mediante clic
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            cachedFile = fileInput.files[0];
            showPreview(fileInput.files[0]); // Muestra vista previa de la imagen
            toggleButtonState(true);
        } else {
            toggleButtonState(false);
        }
    });

    // Función para mostrar la vista previa de la imagen
    function showPreview(file) {
        const reader = new FileReader();
        reader.onload = function(event) {
            const previewImg = document.getElementById('preview-img');
            previewImg.src = event.target.result;
            previewImg.style.display = 'block';

            const img = new Image();
            img.onload = function() {

                if(img.width > img.height){
                    contentOptions.style.display = 'block';
                    optionVerticalImage.style.display = 'none';
                    optionHorizontalImage.style.display = 'block';
                    optionCuadradaImage.style.display = 'none';
                    width_resize = 350;
                    height_resize = 233;
                    val_width.textContent = '350'
                    val_height.textContent = '233'

                }else if(img.width < img.height ){
                    contentOptions.style.display = 'block';
                    optionHorizontalImage.style.display = 'none';
                    optionVerticalImage.style.display = 'block';
                    optionCuadradaImage.style.display = 'none';
                    width_resize = 300;
                    height_resize = 445;
                    val_width.textContent = '300'
                    val_height.textContent = '445'
                }else if(img.width == img.height){
                    contentOptions.style.display = 'block';
                    optionVerticalImage.style.display = 'none';
                    optionHorizontalImage.style.display = 'none';
                    optionCuadradaImage.style.display = 'block';
                    width_resize = 350;
                    height_resize = 350;
                    val_width.textContent = '350'
                    val_height.textContent = '350'
                }
            };
            img.src = event.target.result;
        }
        reader.readAsDataURL(file);
    }

    function toggleButtonState(enabled) {
        button.disabled = !enabled;
    }

    // Enviar la imagen al servidor
    button.addEventListener('click', async () => {
        if(!cachedFile) {
            console.error('No hay archivo cargado.');
            return;
        }

        const formData = new FormData();
        let data;

        formData.append('image', cachedFile);
        formData.append('width',width_resize);
        formData.append('height',height_resize);
        console.log(width_resize);
        console.log(height_resize);

        try{
            const response = await fetch('/resize_image', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Error ${response.status}: ${response.statusText}`);
            }

            const contentType = response.headers.get('Content-Type');
            if (contentType && contentType.includes('application/json')) {
                data = await response.json();
                this.response = data;
                this.filePathName = data.path;
                this.width_resize = data.ancho;
                this.height_resize = data.alto;
                val_width_modal.textContent = data.ancho;
                val_height_modal.textContent = data.alto;

            } else {
                throw new Error('Respuesta no es JSON válido');
            }
        } catch (error) {
            console.error('Error al enviar la imagen:', error);
        }

        const timestamp = new Date().getTime();

        // Mostrar imagen procesada en el modal
        modalImgPreview.src = '';
        modalImgPreview.src = `${data.Imagen_Redimensionada}?t=${timestamp}`;

        this.fileModalImageName = 'resized_'+cachedFile.name;

        // Esperar a que la imagen se cargue para obtener sus dimensiones
        modalImgPreview.onload = () => {
            $('#previewImageModal').modal('show'); // Mostrar el modal
        };
        fileInput.value = '';
    });


    function drawPoints(points, imgW, imgH) {

        drawMiniCards(points);

        const pointContainer = document.getElementById('point-container');
        pointContainer.innerHTML = '';

        pointContainer.style.position = 'absolute';
        pointContainer.style.width = `${imgW}px`;
        pointContainer.style.height = `${imgH}px`;
        points.forEach((point) => {
            const [index, x, y] = point;
            const pointDiv = document.createElement('div');

            pointDiv.classList.add('point');
            pointDiv.setAttribute('data-index', index);
            pointDiv.style.position = 'absolute';

            pointDiv.style.left = `${(x / imgW) * imgW}px`;
            pointDiv.style.top = `${(y / imgH) * imgH}px`;
            pointDiv.style.backgroundColor = '#f03030';
            pointDiv.style.borderRadius = '100%';
            pointDiv.style.border = '1px solid white';
            pointDiv.style.width = '8px';
            pointDiv.style.height = '8px';

            makePointDraggable(pointDiv, index);

            pointContainer.appendChild(pointDiv);

        });
    }

    function drawMiniCards(points) {

        divToPoints.forEach((option) => {
            const card = option.divName;
            if (card) {
                card.style.backgroundColor = 'rgb(255, 105, 105)';
                card.innerHTML = '';
            }
        });
        points.forEach((point) => {
            const [index] = point;

            const matchedOption = divToPoints.find((option) => option.id === index.toString());
            if (matchedOption && matchedOption.divName) {
                const card = matchedOption.divName;

                card.style.backgroundColor = 'rgb(88, 229, 65)';

                card.addEventListener("mouseenter", () => {
                    card.style.cursor = "pointer";
                    card.style.borderRadius = '10px';
                    card.style.boxShadow =  "0px 2px 4px black";

                    //VALOR QUE ASOCIA LA LISTA CON EL ARRAY DE PUNTOS
                    const pointDiv = document.querySelector(`.point[data-index="${index}"]`);
                    if (pointDiv) {
                        pointDiv.style.backgroundColor = '#00f';
                    }
                });

                card.addEventListener("mouseleave", () => {
                    card.style.cursor = "pointer";
                    card.style.borderRadius = '0px';
                    card.style.transition  = '400ms all ease-in-out';
                    card.style.boxShadow = 'none';

                    // Restaurar el color original del punto
                    const pointDiv = document.querySelector(`.point[data-index="${index}"]`);
                    if (pointDiv) {
                        pointDiv.style.backgroundColor = '#f03030'; // Color original
                    }
                });

                const indexParagraph = document.createElement('p');
                indexParagraph.textContent = `${matchedOption.name}`;
                indexParagraph.style.color = 'white';
                indexParagraph.style.fontSize = '12px';
                indexParagraph.style.textAlign = 'center';
                indexParagraph.style.height = '30px';
                indexParagraph.style.lineHeight = '30px';

                card.innerHTML = '';
                card.appendChild(indexParagraph);
            }
        });
    }

    function makePointDraggable(pointDiv, index) {
        pointDiv.onmousedown = function (event) {
            const container = document.getElementById('point-container');
            const containerRect = container.getBoundingClientRect();
    
            const onMouseMove = (e) => {
                let x = e.clientX - containerRect.left;
                let y = e.clientY - containerRect.top;
    
                if (x < 0) x = 0;
                if (y < 0) y = 0;
                if (x > containerRect.width) x = containerRect.width;
                if (y > containerRect.height) y = containerRect.height;
                
                pointDiv.style.left = `${x}px`;
                pointDiv.style.top = `${y}px`;

                const pointIndex = points.findIndex(p => p[0] === index);

                if (pointIndex !== -1) {
                    points[pointIndex][1] = Math.trunc(x);
                    points[pointIndex][2] = Math.trunc(y);
                
                } else {
                    console.error(`No se encontró el índice para index: ${index}`);
                }                
            };

            document.addEventListener('mousemove', onMouseMove);
    
            document.onmouseup = () => {
                document.removeEventListener('mousemove', onMouseMove);
                document.onmouseup = null;
            };
        };
    
        pointDiv.ondragstart = () => false;

    }

    function savePoints(){
        const data = {
            points_position : points,
            file : this.fileModalImageName,
            width : this.width_resize,
            height : this.height_resize,
            pathToSave : selectedOptionPath //
        };
                console.log('data',selectedOptionPath)

        console.log('data',data)

        try{
            const response = fetch('/save', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ data })
            });
        } catch (error) {
            console.error('Error al enviar la imagen:', error);
        }
        closeModalImageEditor()
    }

    async function generatePose(){
        closeModal();
        let position_image;
        const fileName = this.fileModalImageName;

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ fileName })
            });

            if (!response.ok) {
                const errorText = await response.text(); // Leer el texto del error para depurar
                console.error('Error en la respuesta del servidor:', errorText);
                throw new Error('Error en la respuesta del servidor');
            }

            const data = await response.json();
            console.log(data)
            this.response = data;

            if (Array.isArray(this.response.position)) {
                points = this.response.position;

                if(points.length === 0){
                    modal_error.style.display = 'block';
                    buttonSaveImage.disabled = true;
                    buttonSaveImage.style.backgroundColor = '#0067e5';
                    buttonSaveImage.style.opacity = '0.5';

                    let timeLeft = 10;
                    const contador = document.getElementById('cont');
                    const countdown = setInterval(() => {
                        timeLeft--;
                        if (timeLeft >= 0) {
                            contador.textContent = timeLeft;
                        } else {
                            clearInterval(countdown);
                            window.location.reload();
                        }
                    }, 1000);

                }
            } else {
                console.error('El formato de position no es válido:', this.response.position);
                points = [];
            }
            
            // this.points = data.position;
            this.position_image = data.image_pos;

            const timestamp = new Date().getTime();

            modalImg.src = `${this.response.path}?t=${timestamp}`;


        } catch (error) {
            console.error('Error al enviar la imagen:', error);
        }

        // Esperar a que la imagen se cargue para obtener sus dimensiones
        modalImg.onload = () => {
            const imgW = width_resize;
            const imgH = height_resize;

            $('#imageModal').modal('show'); // Mostrar el modal

            drawPoints(this.response.position, imgW, imgH);
        };
    }

    function closeModal(){
        $('#previewImageModal').modal('hide');
    }

    function closeModalImageEditorIcon(){
         modalImg.src = '';
         $('#imageModal').modal('hide');
    }

    function closeModalImageEditor(){
        modalImg.src = '';
        previewImg.src = '';
        previewImg.style.display = 'none';
        $('#imageModal').modal('hide');

        const messageDiv = document.createElement('div');
        messageDiv.textContent = 'Imagen guardada correctamente';
        messageDiv.style.position = 'fixed';
        messageDiv.style.bottom = '20px';
        messageDiv.style.right = '20px';
        messageDiv.style.backgroundColor = '#4caf50';
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

    function openModalImageEditor(){
        $('#imageModal').modal('show');
    }

    function fillSelect(options, selectId) {
        const selectElement = document.getElementById(selectId);

        // Limpia las opciones existentes (si las hay)
        selectElement.innerHTML = '';

        // Itera sobre el array para crear <option> por cada objeto
        options.forEach(option => {
            const optionElement = document.createElement('option');
            optionElement.value = option.id; // Asigna el id como value
            optionElement.textContent = option.name; // Usa el name como texto
            selectElement.appendChild(optionElement);
        });
    }


    //CAMBIO DE OPCIONES EN EL SELECT
    /*$('#imageModal').modal({
        backdrop:'static',
        keyboard:false
    });*/

    document.addEventListener('click', (event) => {
    });

    optionHorizontalImage.addEventListener('click', (event) => {
        event.stopPropagation();
    });
    optionHorizontalImage.addEventListener('change', (event) => {
        this.optionHorizontalImage = event.target.value;
    });

    optionCuadradaImage.addEventListener('click', (event) => {
        event.stopPropagation();
    });
    optionCuadradaImage.addEventListener('change', (event) => {
        this.optionCuadradaImage = event.target.value;
    });



    async function resizeNewDimension() {
        let data;
        let new_image;
        let width, height;

        // if(this.optionHorizontalImage != null){
        //     if (this.optionHorizontalImage === 'option1') {
        //         width = 250;
        //         height = 167;
        //     } else if (this.optionHorizontalImage === 'option2') {
        //         width = 300;
        //         height = 200;
        //     } else if (this.optionHorizontalImage === 'option3') {
        //         width = 350;
        //         height = 233;
        //     } else {
        //         console.error('Opción no válida');
        //         return;
        //     }
        // }

        this.new_image = '';
        const formData = new FormData();
        formData.append('image', this.fileModalImageName);
        formData.append('width', width_resize);
        formData.append('height', height_resize);
        
        console.log('formData',formData)
        
        try {
            const response = await fetch('/resize_image_params', {
                method: 'POST',
                body: formData
            });

            data = await response.json();
            console.log('data',data)

            if (data.path != null) {
                this.new_image = `${data.path}?timestamp=${new Date().getTime()}`;

                if (data.position != null) {
                    drawPoints(data.position, this.width_resize, this.height_resize);
                    this.points = data.position;
                }
            } else {
                console.error('La respuesta no contiene el path de la imagen');
            }

        } catch (error) {
            console.error('Error al enviar la imagen:', error);
        }

        // Cierra el modal y actualiza el contenido visual
        closeModalImageEditor();

        setTimeout(() => {
            openModalImageEditor();
            w.textContent = `${width}px`;
            h.textContent = `${height}px`;
            modalImg.src = this.new_image;
        }, 2000);

    }

    async function cargarRutas() {
        try {
            const selectOptions = document.querySelector("#selectOptions");
            if (!selectOptions) {
                throw new Error("El elemento selectOptions no existe en el DOM.");
            }

            let data = new FormData();
            data.append('id', localStorage.getItem('id'));

            const response = await fetch('/all_paths', {
                method: 'POST',
                body: data
            });

            if (!response.ok) {
                throw new Error("Error al obtener los datos del servidor.");
            }

            const datos = await response.json();
            selectOptions.innerHTML = ""; // Limpia opciones previas
            datos.forEach(fila => {
                const option = document.createElement('option');
                option.value = fila.id;
                option.textContent = fila.nombre;
                selectOptions.appendChild(option);

            });
        } catch (error) {
            console.error('Error al cargar los datos:', error);
        }
    };


