from djitellopy import Tello
from ultralytics import YOLO
import cv2, time

tello = Tello()  # Инициализация дрона
tello.connect()  # Подключение к дрону

print(f'Заряд батареи: {tello.get_battery()}%')
print(f'Температура: {tello.get_temperature()} ℃')

tello.streamon()  # Включение камеры дрона
frame_read = tello.get_frame_read()  # Создание объекта чтения кадров
height, width, _ = frame_read.frame.shape  # Получение разрешения камеры
xcf = width // 2 # Координаты центра кадра
ycf = height // 2
video = cv2.VideoWriter('video_out.avi', cv2.VideoWriter_fourcc(*'XVID'), 1, (width, height))

dist = 0.1833739461042 # Кол-во см в одном пикселе кадра

start = [0, 0]

model = YOLO('cars.pt')  # Инициализация модели машинного обучения

tello.takeoff() # Взлёт дрона
tello.moveup(100) # Подъём дрона до 180 см

while True:
    frame = frame_read.frame  # Получение кадра
    results = model(frame)  # Запись результатов работы модели

    for box in results[0].boxes:  # Перебор обводки каждого распознанного объекта
        if box.conf[0] > 0.5:  # Если значение совпадения больше 60%...
            [x1, y1, x2, y2] = results[0].boxes[0].xyxy[0]  # Получения координат обводки
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)  # Перевод в тип integer
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Запись обводки на кадр с видео
            xc = (x1 + x2) // 2  # Вычисление центра координат обводки по оси X
            yc = (y1 + y2) // 2  # Вычисление центра координат обводки по оси Y

            ycm = (xc - xcf) * dist # Вычисление координаты центра автомобиля-нарушителя относительно центра кадра в сантиметрах по оси X
            xcm = (yc - ycf) * dist # Вычисление координаты центра автомобиля-нарушителя относительно центра кадра в сантиметрах по оси Y

            start = start[0] + xcm, start[1] + ycm # Запись координат относительно точки взлёта

            tello.go_xyz_speed(xcm, ycm, 0, 50) # Выравнивание дрона по направлению к автоиобилю

            print(ycm, xcm) # Вывод координат в консоль

    time.sleep(1)


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
