import pygame
import numpy as np
from numba import prange, njit
import matplotlib.pyplot as plt
import PySimpleGUI as sg

# Параметры визуализации по умолчанию
DEFAULT_WIDTH, DEFAULT_HEIGHT = 600, 600
DEFAULT_MAX_ITER = 200
DEFAULT_MIN_X, DEFAULT_MIN_Y, DEFAULT_MAX_X, DEFAULT_MAX_Y = -1, -1, 1, 1
DEFAULT_R = 2
DEFAULT_COLORMAP = 'twilight_shifted_r'


# ---- Окно настройки параметров ----
def create_settings_window(width: int, height: int, max_iter: int, min_x: float, min_y: float, max_x: float,
                           max_y: float, r: float, colormap: str):
    """
    Создает окно настройки параметров визуализации.

    Parameters:
        width (int): Ширина окна.
        height (int): Высота окна.
        max_iter (int): Максимальное количество итераций.
        min_x (float): Минимальное значение по оси Ox.
        min_y (float): Минимальное значение по оси Oy.
        max_x (float): Максимальное значение по оси Ox.
        max_y (float): Максимальное значение по оси Oy.
        r (float): Радиус сходимости для расчетов.
        colormap (str): Название цветовой карты.

    Returns:
        sg.Window: Окно настроек параметров.
    """
    colormap_options = ['twilight_shifted_r', 'twilight', 'viridis', 'plasma', 'inferno']

    layout = [
        [sg.Text('Width:'), sg.InputText(width)],
        [sg.Text('Height:'), sg.InputText(height)],
        [sg.Text('Max Iterations:'), sg.InputText(max_iter)],
        [sg.Text('Minimal Ox:'), sg.InputText(min_x)],
        [sg.Text('Minimal Oy:'), sg.InputText(min_y)],
        [sg.Text('Maximal Ox:'), sg.InputText(max_x)],
        [sg.Text('Maximal Oy:'), sg.InputText(max_y)],
        [sg.Text('Radius of convergence:'), sg.InputText(r)],
        [sg.Text('Colormap:'), sg.Combo(colormap_options, default_value=colormap, key='colormap')],
        [sg.Button('OK'), sg.Button('Cancel')]
    ]

    window = sg.Window('Settings', layout, finalize=True)
    return window


def get_user_settings():
    """
    Получает параметры от пользователя через окно настроек.

    Returns:
        tuple: Кортеж с параметрами (width, height, max_iter, min_x, min_y, max_x, max_y, r, colormap).
    """
    # Запускаем окно настроек и получаем параметры
    settings_window = create_settings_window(DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_MAX_ITER,
                                             DEFAULT_MIN_X, DEFAULT_MIN_Y, DEFAULT_MAX_X, DEFAULT_MAX_Y, DEFAULT_R,
                                             DEFAULT_COLORMAP)

    while True:
        event, values = settings_window.read()
        if event == sg.WINDOW_CLOSED or event == 'Cancel':
            break
        if event == 'OK':
            width = int(values[0])
            height = int(values[1])
            max_iter = int(values[2])
            min_x = float(values[3])
            min_y = float(values[4])
            max_x = float(values[5])
            max_y = float(values[6])
            r = float(values[7])
            colormap = values['colormap']
            break

    settings_window.close()
    return width, height, max_iter, min_x, min_y, max_x, max_y, r, colormap


# ---- Вычислительная часть ----

# Расчетная функция
@njit
def func(z: complex, c: complex) -> complex:
    """
    Вычисляет значение функции в точке.

    Parameters:
        z (complex): Точка, в которой происходит вычисление.
        c (complex): Комплексная константа (очень важна для множества Жюлиа, кстати).

    Returns:
        complex: Комплексное значение.
    """
    return z ** 2 + c


@njit
def check_point(c: complex, r: float, max_iter: int, z: complex) -> int:
    """
    Проверяет точку на хаотичность.

    Parameters:
        c (complex): Комплексная константа.
        r (float): Радиус сходимости.
        max_iter (int): Максимальное количество итераций.
        z (complex): Проверяемая комплексная точка на хаотичность.

    Returns:
        int: Количество итераций, сделанных для того, чтобы точка ушла на бесконечность. Если оно равно max_it,
    то будем считать, что точка не хаотична.
    """
    res = z
    for i in prange(max_iter):
        if abs(res) > r:
            return i
        res = func(res, c)
    return max_iter


@njit(parallel=True)
def set_calculate(c: complex, min_x: float, max_x: float, min_y: float, max_y: float, r: float, width: int, height: int,
                  max_iter: int) -> np.ndarray:
    """
    Проверяет каждую точку области на хаотичность.

    Parameters:
        c (complex): Комплексная константа.
        min_x: (float): Минимальное значение абсциссы области.
        max_x: (float): Максимальное значение абсциссы области.
        min_y: (float): Минимальное значение ординаты области.
        max_y: (float): Максимальное значение абсциссы области.
        r (float): Радиус сходимости.
        width (int): Ширина окна анимации.
        height (int): Высота окна анимации.
        max_iter (int): Максимальное количество итераций.

    Returns: np.ndarray: Возвращает массив, в котором для каждой точке на комплексной решетке вычислено значение
    функции check_point. Таким образом мы понимаем, как будет выглядeть множество Жюлиа.
    """
    dx = (max_x - min_x) / width
    dy = (max_y - min_y) / height
    result = np.empty((height, width), dtype=np.uint16)
    for iy in prange(height):
        for ix in prange(width):
            x = min_x + ix * dx
            y = min_y + iy * dy
            it = check_point(c, r, max_iter, complex(x, y))
            result[iy, ix] = it
    return result


def create_custom_colormap(colormap_name, max_iter: int):
    """
    Создает пользовательскую цветовую карту.

    Parameters:
        colormap_name (str): Название цветовой карты.
        max_iter (int): Максимальное количество итераций.

    Returns:
        np.ndarray: Массив цветов в формате RGB.
    """
    cmap = plt.cm.get_cmap(colormap_name)
    colors = cmap(np.linspace(0, 1, max_iter))
    custom_cmap = colors[:, :3] * 255  # Оставляем только RGB-компоненты и масштабируем до [0, 255]
    return custom_cmap.astype(np.uint8)


# Инициализация Pygame
pygame.init()
width, height, max_iter, min_x, min_y, max_x, max_y, r, colormap = get_user_settings()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Julia Set Visualization')

running = True
c_real, c_imag = 0.3341, 0.3966
prev_mouse_pos = None

# Добавляем переменные для отслеживания FPS
clock = pygame.time.Clock()
font = pygame.font.Font(None, 36)
fps_text = font.render('FPS: ', True, (255, 255, 255))
complex_point_text = font.render('Complex Point: ', True, (255, 255, 255))

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    x, y = pygame.mouse.get_pos()

    if prev_mouse_pos is None or (x, y) != prev_mouse_pos:
        c_real = min_x + (x / width) * (max_x - min_x)
        c_imag = min_y + ((height - y) / height) * (max_y - min_y)

        c = complex(round(c_real, 4), round(c_imag, 4))
        pi = set_calculate(c, min_x, max_x, min_y, max_y, r, width, height, max_iter)

        # Получаем выбранную цветовую карту из выпадающего меню
        selected_colormap = colormap
        custom_color_map = create_custom_colormap(selected_colormap, max_iter)

        # Применяем цветовую схему
        colored_pi = custom_color_map[pi % max_iter]

        # Создаем изображение
        image = pygame.surfarray.make_surface(colored_pi)

        # Отображаем изображение на экране
        screen.blit(image, (0, 0))

        # Отображаем FPS и текущую комплексную точку
        fps = clock.get_fps()
        fps_text = font.render(f'FPS: {fps:.2f}', True, (255, 255, 255))
        complex_point_text = font.render(f'Complex Point: {c:.4f}', True, (255, 255, 255))

        # Отображаем текст на экране
        screen.blit(fps_text, (10, 10))
        screen.blit(complex_point_text, (10, 50))

        pygame.display.flip()

    prev_mouse_pos = (x, y)

    # Ограничиваем FPS
    clock.tick(144)

# Закрытие окна Pygame при завершении программы
pygame.quit()
