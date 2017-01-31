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
stop_button = False
stop = np.zeros((20, 20, 3), np.uint8)
stop[0:20, 0:20] = (0,0,255)

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

    def stop_screen(self, event, x, y, flags, param ):
        global stop_button, lin
        if (event == cv2.EVENT_LBUTTONDBLCLK) and (stop_button == False):
            stop_button = True


    def execute(self):
        # Add your code here
        global lin, stop, stop_button, stop, refPt, refMov



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
            if key == 13:
                if len(refPt) == 2:
                    cv2.destroyWindow('ROI SELECTION')
                    stop_button = False
                    break
                else:
                    continue
        minRed = np.array((22., 97., 0.))
        maxRed = np.array((48., 153., 255.))
        frame_final = self.camera.getImage()
        frame_final_cut = frame_final[60:420, 0:640]
        frame_final_cut = cv2.medianBlur(frame_final_cut, 3)
        roi = frame_final_cut[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(roi_hsv, minRed, maxRed)
        frame_final_cut_gray = cv2.cvtColor(frame_final_cut, cv2.COLOR_BGR2GRAY)
        backg = np.zeros_like(frame_final_cut_gray)
        backg[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]=mask
        p0 = cv2.goodFeaturesToTrack(backg, 30, 0.01, 10, None, None, 7)
        count = 1
        maxX = 0
        minX = 0
        maxY = 0
        minY = 0
        while (stop_button == False):





            frame_final2 = self.camera.getImage()
            frame_final_cut2 = frame_final2[60:420, 0:640]
            frame_final_cut2 = cv2.medianBlur(frame_final_cut2, 3)
            frame_final_cut2_gray = cv2.cvtColor(frame_final_cut2, cv2.COLOR_BGR2GRAY)
            if p0 is not None:
               if count == 1:
                  maxX = refPt[1][0] + 10
                  minX = refPt[0][0] - 10
                  maxY = refPt[1][1] + 10
                  minY = refPt[0][1] - 10

                  count = 0
               else:
                  maxX = maxX+10
                  minX = minX-10
                  maxY = maxY+10
                  minY = minY-10

               maxX = np.int0(maxX)
               minX = np.int0(minX)
               maxY = np.int0(maxY)
               minY = np.int0(minY)
               if maxX>640:
                   maxX = 640
               if minX <0:
                   minX = 0
               if maxY>320:
                   maxY = 320
               if minY<0:
                   minY = 0
               roi2 = frame_final_cut2[minY:maxY, minX:maxX]
               roi2_hsv = cv2.cvtColor(roi2, cv2.COLOR_RGB2HSV)
               mask2 = cv2.inRange(roi2_hsv, minRed, maxRed)
               backg2 = np.zeros_like(frame_final_cut2_gray)
               backg2[minY:maxY, minX:maxX] = mask2

               p1, st, err = cv2.calcOpticalFlowPyrLK(backg, backg2, p0, None,
                                                   None, None,
                                                   (30, 30), 2,
                                                   (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

               good_p1 = p1[st==1]
               maxAll = np.amax(good_p1, axis = 0)
               minAll = np.amin(good_p1, axis = 0)
               maxX = maxAll[0]#[0]
               maxY = maxAll[1]#[1]
               minX = minAll[0]#[0]
               minY = minAll[1]#[1]

               for i,(f2,f1) in enumerate(zip(p1,p0)):
                  a, b = f2.ravel()
                  c, d = f1.ravel()
                  cv2.circle(frame_final_cut2, (a, b), 5, (255, 255, 0), -1)
                  cv2.circle(frame_final_cut2, (c, d), 5, (255, 0, 0), -1)
                  cv2.line(frame_final_cut2, (a, b), (c, d), (0,0,255), 2)

               cv2.rectangle(frame_final_cut2, (np.int0(minX), np.int0(minY)), (np.int0(maxX), np.int0(maxY)), (0,255,0), 2)

               cv2.imshow("DOUBLE-CLICK STOP BUTTON", stop)
               cv2.setMouseCallback("DOUBLE-CLICK STOP BUTTON", self.stop_screen)
               if frame_final_cut2 is not None:
                   self.camera.setColorImage(frame_final_cut2)

               backg = np.copy(backg2)


               p0 = cv2.goodFeaturesToTrack(backg, 30, 0.01, 10, None, None, 7)
               print("maxX", maxX)
               print("minX", minX)
               print("maxY", maxY)
               print("minY", minY)
               print("----------------------------------")
               print("----------------------------------")
            else:
                if maxX>590:
                    maxX_m = 640
                else:
                    maxX_m = maxX+50
                if minX<50:
                    minX_m = 0
                else:
                    minX_m = minX-50
                if maxY > 270:
                    maxY_m = 320
                else:
                    maxY_m = maxY+50
                if minY<50:
                    minY_m = 0
                else:
                    minY_m = minY-50
                maxX_m = np.int0(maxX_m)
                minX_m = np.int0(minX_m)
                maxY_m = np.int0(maxY_m)
                minY_m = np.int0(minY_m)
                roi3 = frame_final_cut2[minY_m:maxY_m, minX_m:maxX_m]
                roi3_hsv = cv2.cvtColor(roi3, cv2.COLOR_RGB2HSV)
                mask3 = cv2.inRange(roi3_hsv, minRed, maxRed)
                backg = np.zeros_like(frame_final_cut2_gray)
                backg[minY_m:maxY_m, minX_m:maxX_m] = mask3
                p0 = cv2.goodFeaturesToTrack(backg, 30, 0.01, 10, None, None, 7)
                if frame_final_cut2 is not None:
                    self.camera.setColorImage(frame_final_cut2)




        cv2.destroyWindow("DOUBLE-CLICK STOP BUTTON")
        refPt = []
        refMov = []
        lin = np.zeros((360, 640), dtype=np.uint8)





