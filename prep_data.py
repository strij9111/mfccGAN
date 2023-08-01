import os
import random

# путь к вашему каталогу
path = 'data/wavs'

# получить список всех файлов
all_files = os.listdir(path)
all_files = [file for file in all_files if file.endswith('.wav')]

# перемешать список файлов
random.shuffle(all_files)

# определить индекс для разделения на обучающий и валидационный наборы
split_idx = int(0.8 * len(all_files))

# создать обучающий и валидационный списки
training_files = all_files[:split_idx]
validation_files = all_files[split_idx:]

# функция для записи списка файлов в текстовый файл
def write_to_file(file_list, filename):
    with open(filename, 'w') as f:
        for item in file_list:
            f.write("%s\n" % item)

# записать обучающий и валидационный списки в файлы
write_to_file(training_files, 'training.txt')
write_to_file(validation_files, 'validation.txt')
