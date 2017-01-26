import threading
import time
from datetime import datetime
import cv2
import numpy as np

from sensors.cameraFilter import CameraFilter
from parallelIce.navDataClient import NavDataClient
from parallelIce.cmdvel import CMDVel
from parallelIce.extra import Extra
from parallelIce.pose3dClient import Pose3DClient


time_cycle = 80
croppingExt = False
lin = np.zeros((360,640), np.uint8)

class MyAlgorithm(threading.Thread):
    global lin


    def __init__(self, camera, navdata, pose, cmdvel, extra):
        self.camera = camera
        self.navdata = navdata
        self.pose = pose
        self.cmdvel = cmdvel
        self.extra = extra

        self.stop_event = threading.Event()
        self.kill_event = threading.Event()
        self.lock = threading.Lock()
        threading.Thread.__init__(self, args=self.stop_event)


    def run (self):

        self.stop_event.clear()

        while (not self.kill_event.is_set()):
           
            start_time = datetime.now()

            if not self.stop_event.is_set():
                self.execute()

            finish_Time = datetime.now()

            dt = finish_Time - start_time
            ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
            #print (ms)
            if (ms < time_cycle):
                time.sleep((time_cycle - ms) / 1000.0)

    def stop (self):
        self.stop_event.set()

    def play (self):
        if self.is_alive():
            self.stop_event.clear()
        else:
            self.start()

    def kill (self):
        self.kill_event.set()

    def click_and_crop(self, event, x, y, flags, param):
        # referencias del grab a las variables globales
        global refPt, croppingExt, refMov, lin
        # Si el boton izquierdo del raton se pulsa, graba los primeros (x,y) e indica que el corte (cropping) se esta
        # realizando
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt = [(x, y)]
            refMov = [(x, y)]
            croppingExt = True

        # Mira a ver si el boton izquierdo ha dejado de presionarse
        elif event == cv2.EVENT_LBUTTONUP:
            # guarda las coordenadas finales (x ,y) e indica que el corte (cropping) se ha acabado
            refPt.append((x, y))
            croppingExt = False

            # Dentro de este elif dibujo un rectangulo alrededor de la region de interes
            lin = np.zeros((360, 640), dtype=np.uint8)
            cv2.rectangle(lin, refPt[0], refPt[1], 255, 2)


        if (event == cv2.EVENT_MOUSEMOVE) and (croppingExt == True):
            if len(refMov) == 1:
                refMov.append((x, y))
                lin = np.zeros((360, 640), dtype=np.uint8)
                cv2.rectangle(lin, refMov[0], refMov[1], 255, 2)


            elif len(refMov) == 2:
                refMov[1] = ((x, y))
                lin = np.zeros((360, 640), dtype=np.uint8)
                cv2.rectangle(lin, refMov[0], refMov[1], 255, 2)



    def execute(self):
        # Add your code here
        global lin



        frame1 = self.camera.getImage()
        gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray_frame1 = gray_frame1[60:420, 0:640]
        img = cv2.add(gray_frame1, lin)
        cv2.imshow('ROI SELECTION', img)
        cv2.setMouseCallback('ROI SELECTION', self.click_and_crop)


        while (True):
            frame = self.camera.getImage()
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = gray_frame[60:420, 0:640]
            img_tru = cv2.add(gray_frame, lin)
            cv2.imshow('ROI SELECTION', img_tru)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                lin = np.zeros((360, 640), dtype=np.uint8)
            # Tecla enter = 13
            elif key == 13:
                if len(refPt) == 2:
                    cv2.destroyWindow('ROI SELECTION')
                    break
                else:
                    continue

        print("Puntos a dibujar en Color Filter", refPt)


        while (True):
            frame_final = self.camera.getImage()
            frame_final_cut = frame_final[60:420, 0:640]
            lin = np.zeros((360, 640, 3), dtype=np.uint8 )
            cv2.rectangle(lin, refPt[0], refPt[1], (0,255,0), 2)
            frame_tru = cv2.add(frame_final_cut, lin)
            self.camera.setColorImage(frame_tru)
            cv2.imshow("STOP", lin)
            print("sigo en el while 1")
            if key == ord("q"):
                print("presionado  q")
                cv2.destroyAllWindows()
                break
            print("sigo en el while 2")

        print("y vuelta a empezar")


