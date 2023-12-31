# Множество Мандельброта

Этот репозиторий содержит Python-код для вычисления и визуализации множества Мандельброта. Реализованы два варианта: один без ускорения с помощью Numba и другой с использованием Numba для оптимизации производительности.

## Описание

Множество Мандельброта - это захватывающий математический объект, известный своими сложными и визуально потрясающими фрактальными узорами. Этот код позволяет вычислять и визуализировать множество Мандельброта с помощью двух разных реализаций:

1. **Мандельброт (Без Numba):**
   - Функция `mandelbrot_proto` вычисляет множество Мандельброта в стандартном, питоновском стиле без использования дополнительных ускорителей.
   - Принимает входные параметры для задания области построения (`minx`, `maxx`, `miny`, `maxy`), размера сетки (`nx`, `ny`), радиуса убегания (`r`) и максимального количества итераций (`max_it`).
   - Результат вычисления множества Мандельброта возвращается в виде двумерного списка.

2. **Мандельброт (С Numba):**
   - Функция `mandelbrot_njit` использует Numba, компилятор "на лету", чтобы значительно ускорить вычисление множества Мандельброта.
   - Предоставляет параллельную реализацию для дополнительного повышения производительности.
   - Как и версия без Numba, она принимает параметры для области построения, размера сетки, радиуса убегания и максимального количества итераций.
   - Результат вычисления множества Мандельброта возвращается в виде массива NumPy.

## Использование

Вы можете использовать этот код для генерации и визуализации множества Мандельброта, указывая желаемые параметры и параметры отображения в своем скрипте Python. Для правильной работы требуются следующие библиотеки:

- `numpy`
- `matplotlib`
- `numba` (для ускоренной версии с Numba)

Убедитесь, что вы установили необходимые зависимости, если еще этого не сделали. Затем вы можете адаптировать и расширить предоставленный код, чтобы удовлетворить ваши специфические потребности в исследовании и визуализации множества Мандельброта.

## Сравнение производительности

Ускоренная версия с использованием Numba (`mandelbrot_njit`) работает значительно быстрее, чем версия без ускорения (`mandelbrot_proto`). В наших тестах ускоренная версия с Numba выполнялась до 40 раз быстрее, сокращая время вычисления с 800 мс до 20 мс для одной и той же задачи генерации множества Мандельброта.

Кроме того, оптимизация кода (например, замена `range` на `prange`) позволяет добиться еще более высокой производительности, сокращая время выполнения до 8 мс.

Не стесняйтесь экспериментировать с кодом и параметрами, чтобы более детально исследовать множество Мандельброта и оптимизировать его генерацию под ваши конкретные задачи.

## Лицензия

Этот код предоставляется под [Лицензией MIT](LICENSE). Вы можете свободно использовать, модифицировать и распространять его в соответствии с условиями лицензии.

Если у вас есть вопросы или предложения, не стесняйтесь обращаться или создавать запросы в этом репозитории. Наслаждайтесь исследованием удивительного мира множества Мандельброта!