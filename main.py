from djitellopy import Tello
from ultralytics import YOLO

tello = Tello()  # Инициализация дрона
tello.connect()  # Подключение к дрону

print(f'Заряд батареи: {tello.get_battery()}%')
print(f'Температура: {tello.get_temperature()} ℃')

tello.streamon()  # Включение камеры дрона
frame_read = tello.get_frame_read()  # Создание объекта чтения кадров
height, width, _ = frame_read.frame.shape  # Получение разрешения камеры
xcf = width // 2
ycf = height // 2
video = cv2.VideoWriter('video_out_3.avi', cv2.VideoWriter_fourcc(*'XVID'), 15, (width, height))

dist = 0.1833739461042

start = [0, 0]

model = YOLO('car.pt')  # Инициализация модели машинного обучения

tello.takeoff()  # Взлёт дрона
tello.moveup(110)

while True:
    frame = frame_read.frame  # Получение кадра
    results = model(frame)  # Запись результатов работы модели

    for box in results[0].boxes:  # Перебор обводки каждого распознанного объекта
        if box.conf[0] > 0.6:  # Если значение совпадения больше 60%...
            [x1, y1, x2, y2] = box.xyxy[0]  # Получения координат обводки
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Замена координат обводки на координаты
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Запись обводки на кадр с видео

    if len(results[0].boxes) > 0:  # Выполнение условия при наличии распознанного объекта
        [x1, y1, x2, y2] = results[0].boxes[0].xyxy[0]  # Получения координат обводки
        x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)  # Перевод в тип integer
        xc = (x1 + x2) // 2  # Вычисление центра координат обводки по оси X
        yc = (y1 + y2) // 2  # Вычисление центра координат обводки по оси Y

        ycm = (xc - xcf) * dist
        xcm = (yc - ycf) * dist

        start = start[0] + xcm, start[1] + ycm

        tello.go_xyz_speed(xcm, ycm, 0, 50)

        print(ycm, xcm)


    video.write(frame)  # Сохранение кадра с нанесён
    cv2.imshow("drone", frame)  # Вывод видео с каиеры
    key = cv2.waitKey(1) & 0xff
    if key == 27:  # Выход на Escape
        break

tello.go_xyz_speed(-start[0], -start[1], 0, 50)
tello.land()  # Приземление дрона
cv2.destroyAllWindows()  # Закрытие окна вывода видео
video.release()  # Завершение записи
frame_read.stop()  # Завершение чтения кадров
tello.streamoff()  # Завершение видео потока с дрона
tello.end()  # Завершение работы с дроном
