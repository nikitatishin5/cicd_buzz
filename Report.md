# Отчет по лабораторной работе "Генерация последовательностей"

### Коробецкая Анна Александровна, группа М8О-107М-19
Номер в группе: 5, Вариант: 5 ((остаток от деления (5-1) на 6)+1)

> Оценка: 4.5. *Не исследован вопрос переобучения и достаточности обучения и выразительности сети. С архитектурами LSTM/GRU точно было бы возможно достичь более хороших результатов.*

### Цель работы

Научиться генерировать последовательности с помощью рекуррентных нейронных сетей: RNN, LSTM, GRU.

### Используемые входные данные

Для обучения использовался текст программы на языке программирования C++. Источник данных - https://github.com/gameprogcpp/code/blob/master/Chapter01/Game.cpp (файл "progtext1.txt").

### Предварительная обработка входных данных

Для получения последовательности векторов, подаваемых на вход нейросети, были проделаны следующие действия:
 1. Считываем файл и приводим все символы к нижнему регистру:
      filename = 'progtext1.txt'
      raw_text = open(filename, 'r', encoding='utf-8').read()
      proc_text = raw_text.lower()
Убирать пробелы или знаки препинания не требуется, для сохранения оформления программы.

 2. Создаем словарь уникальных символов, содержащихся в progtext1
      chars = sorted(list(set(proc_text)))
      print (chars)
      char_to_int = dict((c, i) for i, c in enumerate(chars))

      int_to_char = dict((i, c) for i, c in enumerate(chars))

 3. Cгенерируем тренировочную выборку. В качестве признаков объекта используем последовательность из 100 символов текста + 1, который нужно будет предсказывать. Для этого пройдемся окном с единичным шагом по всему тексту:
      seq_length = 100
      dataX = []
      dataY = []
      for i in range(0, n_chars - seq_length, 1):
        seq_in = proc_text[i:i + seq_length]
        seq_out = proc_text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
      n_patterns = len(dataX)
      print('Total Patterns: %s' % n_patterns)

Размер тренировочной выборки: 5265

 4. Нормализуем данные и переводим в _one-hot_ представление:
      X = np.reshape(dataX, (n_patterns, seq_length, 1))
      X = X / float(n_vocab)
      y = np_utils.to_categorical(dataY)

### Эксперимент 1: RNN

#### Архитектура сети

rnn_model = Sequential()
rnn_model.add(SimpleRNN(256, input_shape=(X.shape[1], X.shape[2])))
rnn_model.add(Dropout(0.2))
rnn_model.add(Dense(y.shape[1], activation='softmax'))
rnn_model.compile(loss='categorical_crossentropy', optimizer='adam')
rnn_model.fit(X, y, epochs=25, batch_size=100, callbacks=[ModelCheckpoint('rnn-{epoch:02d}.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='min')])

#### Результат

При обучении из 25 эпох ошибка на тренировочных данных снизилась с `3.6412` до `1.9180` - сеть можно "дообучить".
Полученные результаты сохранили оформление в виде програмнмого кода:
<img src="https://github.com/MAILabs-Edu-AI/lab-sequence-generation-hicat242/blob/master/%D0%A0%D0%B5%D0%B7%D1%83%D0%BB%D1%8C%D1%82%D0%B0%D1%82%20RNN.bmp" width="800" height="600">

#### Вывод по данному эксперименту

На мой взгляд генерация текста получилась довольно не плохой: сохранился вид "программы", строки не повторялись, хотя сам по себе текст исходной программы не отличается разнообразием.

### Эксперимент 2: Однослойная LSTM

#### Архитектура сети

lstm_model = Sequential()
lstm_model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(y.shape[1], activation='softmax'))
lstm_model.compile(loss='categorical_crossentropy', optimizer='adam')
lstm_model.fit(X, y, epochs=25, batch_size=100, callbacks=[ModelCheckpoint('lstm-{epoch:02d}.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='min')])

#### Результат

При обучении из 25 эпох ошибка на тренировочных данных снизилась с `3.6566` до `2.3046` - сеть можно "дообучить".
Полученные результаты не сохранили оформление в виде програмнмого кода:
<img src="https://github.com/MAILabs-Edu-AI/lab-sequence-generation-hicat242/blob/master/%D0%A0%D0%B5%D0%B7%D1%83%D0%BB%D1%8C%D1%82%D0%B0%D1%82%20LSTM1.bmp" width="800" height="600">

#### Вывод по данному эксперименту

Сгенерировать текст, чтобы хотя бы сохранился вид "программы", не говоря уже о повторяющихся частях, не удалось.

### Эксперимент 3: Двухслойная LSTM

#### Архитектура сети

double_lstm_model = Sequential()
double_lstm_model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
double_lstm_model.add(Dropout(0.2))
double_lstm_model.add(LSTM(256))
double_lstm_model.add(Dropout(0.2))
double_lstm_model.add(Dense(y.shape[1], activation='softmax'))
double_lstm_model.compile(loss='categorical_crossentropy', optimizer='adam')
double_lstm_model.fit(X, y, epochs=25, batch_size=100, callbacks=[ModelCheckpoint('lstm2-{epoch:02d}.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='min')])

#### Результат

При обучении из 25 эпох ошибка на тренировочных данных снизилась с `3.6283` до `1.9615` - сеть можно "дообучить". Результат лучше ,чем у варианта с одним слоем.
Полученные результаты сохранили оформление в виде програмнмого кода:
<img src="https://github.com/MAILabs-Edu-AI/lab-sequence-generation-hicat242/blob/master/%D0%A0%D0%B5%D0%B7%D1%83%D0%BB%D1%8C%D1%82%D0%B0%D1%82%20LSTM2.bmp" width="800" height="600">

#### Вывод по данному эксперименту

Сгенерировать текст, чтобы хотя бы сохранился вид "программы" - не получилось. Таже видим повторение обшироного куска "программы".

### Эксперимент 4: GRU

#### Архитектура сети

gru_model = Sequential()
gru_model.add(GRU(256, input_shape=(X.shape[1], X.shape[2])))
gru_model.add(Dropout(0.2))
gru_model.add(Dense(y.shape[1], activation='softmax'))
gru_model.compile(loss='categorical_crossentropy', optimizer='adam')
gru_model.fit(X, y, epochs=25, batch_size=100, callbacks= [ModelCheckpoint('gru-{epoch:02d}.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='min')])

#### Результат

При обучении из 25 эпох ошибка на тренировочных данных снизилась с `3.6611` до `1.6433` - сеть можно "дообучить". Наименьший результат потери при 25 эпохах.
Полученные результаты сохранили оформление в виде програмнмого кода:
<img src="https://github.com/MAILabs-Edu-AI/lab-sequence-generation-hicat242/blob/master/%D0%A0%D0%B5%D0%B7%D1%83%D0%BB%D1%8C%D1%82%D0%B0%D1%82%20GRU.bmp" width="800" height="600">

#### Вывод по данному эксперименту

На мой взгляд генерация текста получилась довольно не плохой: сохранился вид "программы", однако есть повторяющиеся фрагменты.

### Выводы

Так как для данных примеров не ставилась задача сохранения логики или понимания функций/операций программы, то наилучшим образом справились RNN и GRU сети, так как была воспроизведена табуляция и тескт получился довольно правдоподобным (переменные, функции, операции). Текст сгенерированный RNN сетью получился самым разнообразным.
Однако, сети LSTM для данной задачи не подошли. Их тексты на выходе ибо повторяли один и тот же фрагмент и терялася формат программы.
