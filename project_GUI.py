from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
import imageio
import cv2
from tools import generate_detections as gdet
from deep_sort import nn_matching
from deep_sort.tracker import Tracker

import red_light

class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)

        self.master = master
        self.pos = []
        self.line = []
        self.rect = []
        self.choice = []
        self.master.title("GUI")
        self.pack(fill=BOTH, expand=1)

        self.counter = 0
        menu = Menu(self.master)
        self.master.config(menu=menu)

        # 顶部菜单
        # 功能编号0，1，2，3
        works = Menu(menu)
        works.add_command(label="Run Red  Light", command=self.choose_red_light)
        works.add_command(label="Speed Check", command=self.choose_speed)
        works.add_command(label="Vehicle Flowrate", command=self.choose_flowrate)
        works.add_command(label="Zebra Crossing", command=self.choose_zebra)
        menu.add_cascade(label="Works", menu=works)

        file = Menu(menu)
        file.add_command(label="Open", command=self.open_file)
        file.add_command(label="Exit", command=self.client_exit)
        menu.add_cascade(label="File",menu=file)

        analyze = Menu(menu)
        analyze.add_command(label="Region of Interest", command=self.regionOfInterest)
        menu.add_cascade(label="Analyze", menu=analyze)

        self.filename = "image/home.jpg"
        self.imgSize = Image.open(self.filename)
        self.tkimage = ImageTk.PhotoImage(self.imgSize)
        self.w, self.h =(2000, 1300)

        self.canvas =Canvas(master =root, width=self.w, height = self.h)
        self.canvas.create_image(20, 20, image=self.tkimage, anchor="nw")
        self.canvas.pack()

    
    def choose_red_light(self):
        self.choice.clear()
        self.choice.append(0)


    def choose_speed(self):
        self.choice.clear()
        self.choice.append(1)


    def choose_flowrate(self):
        self.choice.clear()
        self.choice.append(2)


    def choose_zebra(self):
        self.choice.clear()
        self.choice.append(3)


    def open_file(self):
        self.filename = filedialog.askopenfilename()

        cap = cv2.VideoCapture(self.filename)
        
        reader = imageio.get_reader(self.filename)
        fps = reader.get_meta_data()['fps']

        ret, image = cap.read()
        cv2.imwrite("image/preview.jpg", image)

        self.show_image("image/preview.jpg")
    

    def show_image(self, frame):
        self.imgSize = Image.open(frame)
        self.tkimage = ImageTk.PhotoImage(self.imgSize)
        self.w, self.h = (2000, 1300)

        self.canvas.destroy()

        self.canvas = Canvas(master=root, width=self.w, height=self.h)
        self.canvas.create_image(0, 0, image=self.tkimage, anchor="nw")
        self.canvas.pack()


    def regionOfInterest(self):
        root.config(cursor="plus")
        self.canvas.bind("<Button-1>", self.imgClick)


    def client_exit(self):
        exit()

    
    def imgClick(self, event):
        if self.choice[0] == 0:

            if self.counter < 4:
                x = int(self.canvas.canvasx(event.x))
                y = int(self.canvas.canvasy(event.y))
                self.line.append((x, y))
                self.pos.append(self.canvas.create_line(x - 5, y, x + 5, y, fill="red", tags="crosshair"))
                self.pos.append(self.canvas.create_line(x - 5, y, x + 5, y, fill="red", tags="crosshair"))
                self.counter += 1

            #if self.counter == 2:
                #unbinding action with mouse-click
                #self.canvas.unbind("<Button-1>")
                #root.config(cursor="arrow")
                #self.counter = 0

                #show created virtual line
                #print(self.line)
                #print(self.rect)
                #img = cv2.imread('image/preview.jpg')
                #cv2.line(img, self.line[0], self.line[1], (0, 255, 0), 3)
                #cv2.imwrite('image/copy.jpg', img)
                #self.show_image('image/copy.jpg')
            if self.counter == 4:
                #unbinding action with mouse-click
                self.canvas.unbind("<Button-1>")
                root.config(cursor="arrow")
                self.counter = 0

                #show created virtual line
                print(self.line)
                print(self.rect)
                img = cv2.imread('image/preview.jpg')
                cv2.line(img, self.line[0], self.line[1], (0, 255, 0), 3)
                cv2.line(img, self.line[2], self.line[3], (0, 255, 0), 3)
                cv2.imwrite('image/copy.jpg', img)
                self.show_image('image/copy.jpg')

                self.red_light_start()
                self.line.clear()
                self.rect.clear()
                for i in self.pos:
                    self.canvas.delete(i)


    def red_light_start(self):
        yolo = red_light.YOLO()
        yolo.set_line(self.line)

        #max_cosine_distance = 0.3
        #nn_budget = None
        #nms_max_overlap = 1.0


        video_src = self.filename
        output = 'image/output.avi'

        # 目标追踪
        model_filename = 'model_data/mars-small128.pb'
        encoder = gdet.create_box_encoder(model_filename,batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric(
            		"cosine", yolo.max_cosine_distance, yolo.nn_budget)
        tracker = Tracker(metric)

        cap = cv2.VideoCapture(video_src)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 1)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(output, fourcc, fps, size)
        ret = True
        frame_index = -1
        while ret :
            ret, frame = cap.read()
            if not ret :
                print('结束')
                break
            image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            image = yolo.detect_image(image,'pic', encoder, tracker)
            cv2.imshow('result', image)
            out.write(image)

        cap.release()
        out.release()








root = Tk()
app = Window(root)
root.geometry("%dx%d"%(1000, 700))
root.title("Traffic System")
root.mainloop()