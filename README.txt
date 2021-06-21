Sebastián Parra
Facultad de Ciencias Físicas y Matemáticas, Universidad de Chile

=======================================================================================
LEER ESTAS INSTRUCCIONES ANTES DE EJECUTAR CUALQUIER ARCHIVO
=======================================================================================

Este proyecto consiste de 3 archivos y 1 carpeta:

- createExamples.cpp: Este es un script que, a partir de un conjunto de imágenes de objetos
	segmentados y un conjunto de imágenes de fondos, generará una cantidad determinada de
	ejemplos para ser utilizados en el entrenamiento de YOLO.

- augmentImages.py: Este es un script que, a partir de un conjunto de ejemplos generados con
	el script anterior, le aplica una secuencia aleatoria de transformaciones a cada una
	de las imágenes para generar 5 copias modificadas de cada ejemplo, con el objetivo de
	realizar augmentation de la base de datos.

- write.py: Este script genera un archivo train.txt, en el cual se especifican todos los archivos
	que debe usar YOLO para el entrenamiento

- darknet (Carpeta): En esta carpeta se encuentran los archivos yolov3-tiny-obj.cfg, obj.data,
	obj.names y train.txt que se usaron para entenar la red YOLO. Además, también se puede
	encontrar un archivo .weights, que corresponde al modelo final elegido en este proyecto.
	Para mayor información sobre cómo configurar YOLO para la detección de objetos, leer las
	instrucciones proporcionadas por AlexeyAB en su repositorio de GitHub:
	https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects


REQUISITOS:
- Versión compilada de darknet de AlexeyAB: https://github.com/AlexeyAB/darknet
- OpenCV3+
- CMake 3.5.1+
- imgaug (libreria de Python: https://github.com/aleju/imgaug)
- Python3
- Numpy
- Pandas
- tqdm


CÓMO HACER BUILD:

1) Crear una nueva carpeta llamada "build" en el directorio de este proyecto

2) Abrir una nueva terminal y navegar a la carpeta creada

3) En la terminar, ejecutar el comando:
	$ cmake ../

4) Finalmente, ejecutar:
	$ make

5) Si todo funcionó correctamente, el archivo ejecutable createExamples deberia haberse creado en la carpeta "build"


CÓMO EJECUTAR:

- createExamples.cpp: 
	Una vez creado el ejecutable, éste puede ser utilizado mediante el comando "$ ./createExamples" en la terminal,
	suponiendo que ésta se encuentra en la carpeta "build". Sin embargo, esto sólo imprimirá una documentación de ayuda
	al usuario. Para ejecutar el código es necesario especificar los siguientes parámetros posicionales:

	1) nExamples: Número de ejemplos que se desea generar

	2) minObjPerImg: Número mínimo de objetos que debe haber en cada ejemplo (puede ser cero)

	3) maxObjPerImg: Número máximo de objetos que puede haber en un ejemplo

	4) objFolder: Carpeta en donde se encuentran las fotos de objetos segmentados (SOLO DEBEN ESTAR ESTAS FOTOS EN LA CARPETA)

	5) bgFolder: Carpeta donde se encuentras las imágenes de fondos (SOLO DEBEN ESTAR ESTAS FOTOS EN LA CARPETA)

	6) outFolder: Carpeta donde se guardarán los ejemplos generados


- augmentImages.py:
	Antes de ejecutar este archivo, es necesario revisar ciertas variables definidas en el código:

	1) IMG_DIR: Directorio genérico donde se encuentran guardados los ejemplos originales generados. 
		Debe ser de la forma "ubicacion/de/la/carpeta/example_{}.png"

	2) BBOX_DIR: Directorio genérico donde se encuentran guardados los archivos .txt que acompañan a
		cada imagen. Debe ser de la forma "ubicacion/de/la/carpeta/example_{}.txt"

	3) AUG_IMG_DIR: Directorio genérico donde se guardarán las imágenes generadas por augmentation.
		Debe ser de la forma "ubicacion/de/la/carpeta/aug_example_{}.png"

	4) AUG_BBOX_DIR: Directorio genérico donde se guardarán los archivos .txt que acompañan a las
		imágenes generadas por augmentation. Debe ser de la forma "ubicacion/de/la/carpeta/aug_example_{}.txt"

	5) N_EXAMPLES: Número de ejemplos originales que existen en la carpeta IMG_DIR

- write.py:
	Al igual que con el archivo anterior, es necesario definir unos parámetros:

	1) EXAMPLES_DIR: Directorio genérico dentro de darknet donde se guardarán los ejemplos originales (no augmentation).
		Debe ser de la forma "build/darknet/x64/ubicacion/de/la/carpeta/example_{}.png"

	2) AUG_DIR: Directorio genérico dentro de darknet donde se guardarán los ejemplos generados por augmentation.
		Debe ser de la forma "build/darknet/x64/ubicacion/de/la/carpeta/example_{}.png"

	3) N_EXAMPLES: Número de ejemplos originales (no augmentation) que se crearon


CÓMO REALIZAR PREDICCIONES CON TINY-YOLO:
Asumiendo que se cumplen los requisitos solicitados en este README, para realizar predicciones con el modelo entrenado
para este proyecto basta con copiar el archivo .cfg proporcionado dentro de la carpeta "build/darknet/x64/cfg/", y los
archivos "obj.data" y "obj.names" dentro de la carpeta "build/darknet/x64/data/", como se especifica en
https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects. 

Luego, se debe copiar el archivo yolov3-tiny-obj_final.weights proporcionado en este proyecto al directorio "darknet/backup"

Una vez cumplido este paso, se debe abrir una terminal en la carpeta "darknet" (NO "darknet/build/darknet") y ejecutar:
	$ ./darknet detector test build/darknet/x64/data/obj.data build/darknet/x64/cfg/yolov3-tiny-obj.cfg backup/yolov3-tiny-		 obj_final.weights /ubicacion/de/imagen

Donde en "/ubicacion/de/imagen" se debe colocar el directorio donde se encuentra la imagen que se desea procesar.




