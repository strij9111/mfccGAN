# mfccGAN
Реализация обратного преобразования MFCC в звук посредством GAN

### Генератор MFCCGAN
Стек транспонированных сверточных слоев используется для увеличения
входного MFCC с нижним временным разрешением до сырой речевой волны. За каждым сверточным слоем располагается блок остаточных
слоев с атрибутом дилатации. Дилатация приводит к большему рецептивному полю входного сигнала, что лучше моделирует вариации

Увеличение разрешения применяется на четырех этапах: два увеличения разрешения 8x и два увеличения 2x. 

На каждом этапе применяется оператор транспонированной свертки 1D для увеличения разрешения. 

Остаточные стеки не меняют размер входного сигнала. Каждый остаточный стек состоит из трех сверточных слоев с различными параметрами дилатации.


### Дискриминатор
Имеет многоуровневую архитектуру, содержащую три идентичных дискриминатора,
которые работают на разных аудиошкалах (частотах дискретизации).

Понижение разрешения сигнала до более низкого разрешения позволяет рассматривать сигнал с различными частотными разрешениями.

Поведение более высокой частоты больше проявляется на более высоких шкалах, в то время как более низкая частота дискретизации
может отражать поведение низкой частоты. Таким образом, масштабирование выходного сигнала на три уровня может привести к лучшему суждению о качестве.

### Функции потерь
#### MCD (Mel Cepstral Distortion): Используется как мера дисторсии между исходным и сгенерированным звуковыми сигналами
#### STOI (Short-Time Objective Intelligibility): используется как оценка качества речи

Обучал на RTX 3090 24gb используя датасет Mozilla Common Voice. Для обучения необходимо разместить файлы из датасета в каталоге data/wavs
и запустить prep_data.py для формирования списков файлов.

## References
[1] M. R. Hasanabadi, M. Behdad, D. Gharavian. (2023). [MFCCGAN: A Novel MFCC-Based Speech Synthesizer Using Adversarial Learning.](https://arxiv.org/abs/2306.12785) arXiv preprint arXiv:2306.12785.
