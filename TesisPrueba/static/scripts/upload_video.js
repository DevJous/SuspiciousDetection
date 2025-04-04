const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const buttonGenerateImagesFromVideo = document.getElementById('button');
const modelImages = document.getElementById('imageModal');
const loader = document.getElementById('loader_container');
const saveButton = document.getElementById('saveButton');
const optionVerticalImage = document.getElementById('select-option-vertical');
const optionHorizontalImage = document.getElementById('select-option-horizontal');
const optionCuadradaImage = document.getElementById('select-option-cuadrada');
const close_icon = document.getElementById('closeIcon');
const selectFrames = document.getElementById('frames');
const modalImage = document.getElementById('modal-img');
const txtNumberFPS = document.getElementById('txtNumberFPS');
const omitirImagen = document.getElementById('btn_omitir_imagen');
const contentOptions = document.getElementById('content-options-elements');
const selectOptions = document.getElementById('selectOptions');

const val_width = document.getElementById('val_width');
const val_height = document.getElementById('val_height');


let points = [];
let fileName = "";
let currentIndex = 0;
let imagesArray = [];
let opcionActual = '';
var width_resize;
var height_resize;
var selectedOptionPath = '';

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

    document.addEventListener('DOMContentLoaded', async () => {
        const nameCache = localStorage.getItem('user');
        let width, height;
        let frame = localStorage.getItem('frames');

        const token = localStorage.getItem('token');
        setTimeout(() => {
            if (!token) {
                window.location.href = '/login';
                return;
            }
        }, 100);

        if(nameCache){
            name.textContent = nameCache;
        }
        await cargarRutas();

        if (selectOptions.options.length > 0) {
            selectOptions.selectedIndex = 0;
            //selectedOption = selectOptions.options[0].textContent;
            selectedOptionPath = selectOptions.options[0].textContent;
            console.log(selectedOptionPath)
        }
        selectOptions.addEventListener('change', (event) => {
            selectedOptionPath = event.target.selectedOptions[0].textContent;
            console.log(selectedOptionPath)
        });

        txtNumberFPS.disabled = true;
        txtNumberFPS.placeholder = localStorage.getItem('frames');

        const modalImage = document.getElementById('modal-img');

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

// Permite arrastrar el archivo sobre la zona de drop
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
        buttonGenerateImagesFromVideo.style.opacity = '1' ;

        // Obtiene el archivo y lo asigna al input
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('video/')) {
            fileInput.files = e.dataTransfer.files; // Asignar al input
            showPreviewVideo(file); // Mostrar la imagen
        }
    });

    dropZone.addEventListener('click', () => {
        e.stopPropagation();
        fileInput.click();
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            showPreview(fileInput.files[0]);
        }
    });

    function showPreviewVideo(file) {
        const videoPreview = document.getElementById('video-preview');
        const videoURL = URL.createObjectURL(file);

        let videoElement = document.createElement('video');
        videoElement.controls = true;
        videoElement.src = videoURL;

        // Agrega la clase para limitar el alto
        videoElement.classList.add('video-preview-video');

        videoPreview.innerHTML = '';
        videoPreview.appendChild(videoElement);
        button.style.display = 'block';

        videoElement.addEventListener('loadedmetadata', function () {
            const videoWidth = videoElement.videoWidth;
            const videoHeight = videoElement.videoHeight;

            console.log('Ancho del video:', videoWidth);
            console.log('Alto del video:', videoHeight);

            if (videoWidth > videoHeight) {
                contentOptions.style.display = 'block';
                optionVerticalImage.style.display = 'none';
                optionHorizontalImage.style.display = 'block';
                optionCuadradaImage.style.display = 'none';
                width_resize = 350;
                height_resize = 233;
                val_width.textContent = '350';
                val_height.textContent = '233';
            } else if (videoWidth < videoHeight) {
                contentOptions.style.display = 'block';
                optionHorizontalImage.style.display = 'none';
                optionVerticalImage.style.display = 'block';
                optionCuadradaImage.style.display = 'none';
                width_resize = 300;
                height_resize = 445;
                val_width.textContent = '300';
                val_height.textContent = '445';
            } else if (videoWidth == videoHeight) {
                contentOptions.style.display = 'block';
                optionVerticalImage.style.display = 'none';
                optionHorizontalImage.style.display = 'none';
                optionCuadradaImage.style.display = 'block';
                width_resize = 350;
                height_resize = 350;
                val_width.textContent = '350';
                val_height.textContent = '350';
            }
        });
    }

    async function GenerateImagesFromVideo() {
        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append('video', file);
        formData.append('fps_value',localStorage.getItem('frames'))


        loader.style.display = 'block';

        try {
            const response = await fetch('/generate_images_from_videos', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Error en la solicitud: ${response.statusText}`);
            }

            const data = await response.json();
            imagesArray = Object.keys(data).map((key) => data[key]);

            const modal = new bootstrap.Modal(document.getElementById('imageModal'));

            loader.style.display = 'none';
            modal.show();

            const previewContainer = document.getElementById('modal_footer_images');
            previewContainer.innerHTML = '';

            imagesArray.forEach((imageData) => {
                const img = document.createElement('img');
                img.src = `data:image/jpeg;base64,${imageData}`;
                img.style.maxWidth = '50px';
                img.style.maxHeight = '50px';
                img.style.margin = '10px';
                previewContainer.appendChild(img);
            });

            const updatePreviewContainer = () => {
                previewContainer.innerHTML = '';
                imagesArray.forEach((imageData) => {
                    const img = document.createElement('img');
                    img.src = `data:image/jpeg;base64,${imageData}`;
                    img.style.maxWidth = '50px';
                    img.style.maxHeight = '50px';
                    img.style.margin = '10px';
                    previewContainer.appendChild(img);
                });
            };

            if (imagesArray.length > 0) {
                const currentImageData = imagesArray[0];

                // Aquí procesas la imagen actual, por ejemplo:
                generatePoseFromBlob(currentImageData);

                omitirImagen.addEventListener('click', ()=>{
                    imagesArray.shift();
                    updatePreviewContainer();
                    if (imagesArray.length > 0) {
                        generatePoseFromBlob(imagesArray[0]);
                    } else {
                        alert("Todas las imágenes han sido procesadas y guardadas.");
                        modelImages.style.display = 'none';
                        modelImages.style.backgroundColor = 'transparent';
                        const backdrop = document.querySelector('.modal-backdrop');
                        if (backdrop) {
                          backdrop.remove();
                        }
                    }
                });

                saveButton.addEventListener('click', async () => {
                    const data = {
                        'points_position': points,
                        'file': fileName,
                        'width': width_resize,
                        'height': height_resize,
                        'pathToSave': selectedOptionPath
                    };
                    console.log(data)

                    modal.hide();
                    try {
                        const response = await fetch('/save_image_from_video', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({ data })
                        });

                        imagesArray.shift();
                        updatePreviewContainer();

                        if (imagesArray.length > 0) {
                            generatePoseFromBlob(imagesArray[0]);
                        } else {
                            alert("Todas las imágenes han sido procesadas y guardadas.");
                        }
                    } catch (error) {
                        console.error('Error al enviar la imagen:', error);
                    } finally {
                        setTimeout(() => modal.show(), 1000);
                    }
                });
            } else {
                alert("No hay más imágenes para procesar.");
            }
        } catch (error) {
            console.error('Error al enviar el video:', error);
        }
    }

    async function generatePoseFromBlob(imageBase64) {
        try {
            const blob = await fetch(`data:image/jpeg;base64,${imageBase64}`).then((res) => res.blob());

            const formData = new FormData();
            formData.append('image', blob, 'image.jpg');
            formData.append('width',width_resize);
            formData.append('height',height_resize);

            const response = await fetch('/upload_image_video', {
                method: 'POST',
                body: formData,
            });
            if (!response.ok) {
                throw new Error(`Error en la solicitud: ${response.statusText}`);
            }

            const result = await response.json();
            points = result.position;
            width_resize  = result.width;
            height_resize = result.height;
            val_width.textContent = result.width;
            val_height.textContent = result.height;
            fileName = result.filename;
            console.log('result',result)
            if (result.path) {
                const modalImage = document.getElementById('modal-img');

                modalImage.src = '';
                modalImage.style.width = `${result.width}`;
                modalImage.style.height = `${result.height}`;

                modalImage.src = `${result.path}?${Math.random()}`;
            }
            drawPoints(points, result.width, result.height);


        } catch (error) {
            console.error('Error al enviar la imagen para generar la pose:', error);
        }
    }

    function drawPoints(points, imgW, imgH) {
        drawMiniCards(points);

        const pointContainer = document.getElementById('point-container');
        pointContainer.innerHTML = '';

        // Configurar el contenedor
        pointContainer.style.position = 'absolute';
        pointContainer.style.width = `${imgW}px`;
        pointContainer.style.height = `${imgH}px`;

        //console.log("Puntos recibidos:", points);

        points.forEach((point) => {
            const [index, x, y] = point;
            console.log(`Punto ${index}:`, point);

            const pointDiv = document.createElement('div');

            pointDiv.classList.add('point');
            pointDiv.setAttribute('data-index', index);
            pointDiv.style.position = 'absolute';

            // Ajustar las posiciones al tamaño del contenedor
            const normalizedX = Math.min(Math.max(x, 0), imgW); // Mantener dentro de [0, imgW]
            const normalizedY = Math.min(Math.max(y, 0), imgH); // Mantener dentro de [0, imgH]

            pointDiv.style.left = `${normalizedX}px`;
            pointDiv.style.top = `${normalizedY}px`;


            // Estilo del punto
            pointDiv.style.backgroundColor = '#f03030';
            pointDiv.style.borderRadius = '50%'; // Para círculos
            pointDiv.style.border = '1px solid white';
            pointDiv.style.width = '8px';
            pointDiv.style.height = '8px';

            // Hacer el punto arrastrable
            makePointDraggable(pointDiv, index);

            // Agregar el punto al contenedor
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

            //console.log("ARRAY FINAL:",points)

            document.addEventListener('mousemove', onMouseMove);

            document.onmouseup = () => {
                document.removeEventListener('mousemove', onMouseMove);
                document.onmouseup = null;
            };
        };
        pointDiv.ondragstart = () => false;
    }

    function closeModalImageEditor(){
        $('#imageModal').modal('hide');
        $(document).ready(function() {
            $('#video-preview').empty();
        });
        buttonGenerateImagesFromVideo.style.opacity = '0' ;
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
