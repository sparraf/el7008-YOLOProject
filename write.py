# Directorio donde se colocarán los ejemplos (debe ser una carpeta dentro de 'darknet')
EXAMPLES_DIR = 'build/darknet/x64/data/obj/example_{}.png' 
# Directorio donde se colocarán los datos generados por augmentation (debe ser una carpeta dentro de 'darknet')
AUG_DIR = 'build/darknet/x64/data/obj/aug_example_{}.png'

# Numero de ejemplos originales creados (no cuenta augmentation)
N_EXAMPLES = 1500

with open('train.txt', 'w') as file:
	for i in range(N_EXAMPLES * 5):
		if i < N_EXAMPLES:
			file.write(EXAMPLES_DIR.format(i))
			file.write('\n')
		file.write(aug_dir.format(i))
		file.write('\n')