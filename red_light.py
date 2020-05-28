#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""
 
import colorsys
import os
import random
import warnings
import cv2
import collections
from timeit import default_timer as timer
import time
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from keras.utils import multi_gpu_model
from PIL import Image, ImageFont, ImageDraw
from PIL import Image
from PIL import ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

import plate_detect.HyperLPRLite as pr

warnings.filterwarnings('ignore') 
fontC = ImageFont.truetype("plate_detect/Font/platech.ttf", 14, 0)
cars_run_line_1 = []
cars_run_line_2 = []
car_dic = {}

class YOLO(object):
    def __init__(self):
        self.model_path = 'model_data/yolo.h5'
        self.anchors_path = 'model_data/yolo_anchors.txt'
        self.classes_path = 'model_data/coco_classes.txt'
        self.score = 0.5
        self.iou = 0.5
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (416, 416) # fixed size or (None, None)
        self.is_fixed_size = self.model_image_size != (None, None)
        self.boxes, self.scores, self.classes = self.generate()
                                                                             #881
        self.line = [(450,700), (1600, 720), (881, 261), (1200, 260)]
        #self.line = [ (1600, 720),(450,700), (1200, 260), (881, 261) ]
        self.straight = True
        self.left = True
        self.right = True
        self.max_cosine_distance = 0.3
        self.nn_budget = None
        self.nms_max_overlap = 1.0

 
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names
 
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors
 
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'
 
        self.yolo_model = load_model(model_path, compile=False)
        print('{} model, anchors, and classes loaded.'.format(model_path))
 
        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.
 
        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes


    def set_line(self, lines):
        self.line[0] = lines[0]
        self.line[1] = lines[1]
        self.line[2] = lines[2]
        self.line[3] = lines[3]


    def getColorList(self):
        dict = collections.defaultdict(list)

        # 红色
        lower_red = np.array([156, 43, 46])
        upper_red = np.array([180, 255, 255])
        color_list = []
        color_list.append(lower_red)
        color_list.append(upper_red)
        dict['red'] = color_list

        # 红色2
        lower_red = np.array([0, 43, 46])
        upper_red = np.array([10, 255, 255])
        color_list = []
        color_list.append(lower_red)
        color_list.append(upper_red)
        dict['red2'] = color_list

        # 橙色
        lower_orange = np.array([11, 43, 46])
        upper_orange = np.array([25, 255, 255])
        color_list = []
        color_list.append(lower_orange)
        color_list.append(upper_orange)
        dict['orange'] = color_list

        # 黄色
        lower_yellow = np.array([26, 43, 46])
        upper_yellow = np.array([34, 255, 255])
        color_list = []
        color_list.append(lower_yellow)
        color_list.append(upper_yellow)
        dict['yellow'] = color_list

        # 绿色
        lower_green = np.array([35, 43, 46])
        upper_green = np.array([77, 255, 255])
        color_list = []
        color_list.append(lower_green)
        color_list.append(upper_green)
        dict['green'] = color_list
        return dict

    def get_color(self,frame):
        print('go in get_color')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        maxsum = -100
        color = None
        color_dict = self.getColorList()
        score = 0
        type = 'black'
        for d in color_dict:
            mask = cv2.inRange(hsv, color_dict[d][0], color_dict[d][1])
            # print(cv2.inRange(hsv, color_dict[d][0], color_dict[d][1]))
            #cv2.imwrite('images/triffic/' + f + d + '.jpg', mask)
            binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
            binary = cv2.dilate(binary, None, iterations=2)
            #img, cnts, hiera = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts, hiera = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sum = 0
            for c in cnts:
                sum += cv2.contourArea(c)

            if sum > maxsum:
                maxsum = sum
                color = d
            if sum > score:
                score = sum
                type = d
        return type

    def recognize_plate(self, image, smallest_confidence = 0.7):
        # # grr = cv2.imread(image_path)

        model = pr.LPR("plate_detect/model/cascade.xml", "plate_detect/model/model12.h5", "plate_detect/model/ocr_plate_all_gru.h5")
        model.SimpleRecognizePlateByE2E(image)
        return_all_plate = []
        for pstr,confidence,rect in model.SimpleRecognizePlateByE2E(image):
            if confidence>smallest_confidence:
                return_all_plate.append([pstr,confidence,rect])
        return return_all_plate

    def drawRectBox(self,image,rect,addText):
        cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), (0,0, 255), 2,cv2.LINE_AA)
        cv2.rectangle(image, (int(rect[0]-1), int(rect[1])-16), (int(rect[0] + 115), int(rect[1])), (0, 0, 255), -1,
                      cv2.LINE_AA)
        img = Image.fromarray(image)
        draw = ImageDraw.Draw(img)
        draw.text((int(rect[0]+1), int(rect[1]-16)), addText.encode('utf-8').decode("utf-8"), (255, 255, 255), font=fontC)
        imagex = np.array(img)
        return imagex

    def visual_draw_position(self,grr):
        model = pr.LPR("plate_detect/model/cascade.xml","plate_detect/model/model12.h5","plate_detect/model/ocr_plate_all_gru.h5")
        for pstr,confidence,rect in model.SimpleRecognizePlateByE2E(grr):
            print(pstr)
            if confidence>0.7:
                grr = self.drawRectBox(grr, rect, pstr+" "+str(round(confidence,3)))
                print("车牌号:")
                print(pstr)
                print("置信度")
                print(confidence)
            cv2.imwrite("image/run_red_light/"+pstr+".jpg", grr)
        # cv2.imshow("image",grr)
        # cv2.waitKey(0)

    def cross(self, p1, p2 , p3):
        x1 = p2[0] - p1[0]
        y1 = p2[1] - p1[1]
        x2 = p3[0] - p1[0]
        y2 = p3[1] - p1[1]
        return x1*y2-x2*y1

    def segment(self, p1, p2, p3, p4):
        if(max(p1[0], p2[0]) >= min(p3[0], p4[0])\
            and max(p3[0],p4[0]) >= min(p1[0],p2[0])\
            and max(p1[1],p2[1]) >= min(p3[1],p4[1])\
            and max(p3[1],p4[1]) >= min(p1[1],p2[1])):
            if(self.cross(p1,p2,p3)*self.cross(p1,p2,p4)<=0\
                and self.cross(p3,p4,p1)*self.cross(p3,p4,p2)<=0):
                return True
            else:
                return False
        else:
            return False

    def intersection(self,line,top,left,bottom,right, num):
        if(line[num][0]>=left and line[num][1]>=top and line[num][0] <= right and line[num][1]<=bottom)or\
        (line[num+1][0]>=left and line[num+1][1]>=top and line[num+1][0]<=right and line[num+1][1]<=bottom):
              return True
        else:
            p1 = [left, top]
            p2 = [right, bottom]
            p3 = [right, top]
            p4 = [left, bottom]
            if self.segment(line[num],line[num+1],p1,p2) or self.segment(line[num],line[num+1],p3,p4):
                return True
            else:
                return False


    def car_tracker(self, boxs, imgcv, encoder, tracker):
        global cars_run_line_1
        global cars_run_line_2
        global car_dic
        features = encoder(imgcv,boxs)
        detections = [Detection(bbox, 1.0, feature) for
              bbox, feature in zip(boxs, features)]
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            '''
            cv2.rectangle(imgcv, 
                                  (int(bbox[0]), int(bbox[1])), 
                                  (int(bbox[2]), int(bbox[3])),
                                  (255,255,255), 
                                  2)
                    '''
                    
            if self.straight==False and self.intersection(self.line,int(bbox[1]),int(bbox[0]),int(bbox[3]),int(bbox[2]), 0):
                if str(track.track_id) not in cars_run_line_1:
                    print(str(track.track_id),"not in cars1:",cars_run_line_1)
                    cars_run_line_1.append(str(track.track_id))
                    if str(track.track_id) not in car_dic.keys():
                        cimg = imgcv[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
                        car_dic.update({str(track.track_id):cimg})
                    
                if (str(track.track_id) in cars_run_line_1) and (str(track.track_id) in cars_run_line_2):
                    print("key:",str(track.track_id))
                    self.visual_draw_position(car_dic[str(track.track_id)])
                    cars_run_line_1.remove(str(track.track_id))
                    cars_run_line_2.remove(str(track.track_id))
#                    cimg = imgcv[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
                    
                cv2.rectangle(imgcv,
                              (int(bbox[0]), int(bbox[1])),
                              (int(bbox[2]), int(bbox[3])),
                              (0,255,0),
                              lineType=2, thickness=8)

                #cimg = imgcv[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
                #self.visual_draw_position(cimg)            
            elif self.straight==False and self.intersection(self.line,int(bbox[1]),int(bbox[0]),int(bbox[3]),int(bbox[2]), 2):
                if str(track.track_id) not in cars_run_line_2:
                    print(str(track.track_id), "not in cars2:",cars_run_line_2)
                    cars_run_line_2.append(str(track.track_id))
                    if str(track.track_id) not in car_dic.keys():
                        cimg = imgcv[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
                        car_dic.update({str(track.track_id):cimg})
                    
                if (str(track.track_id) in cars_run_line_1) and (str(track.track_id) in cars_run_line_2) :
                    print("key:",str(track.track_id))
                    self.visual_draw_position(car_dic[str(track.track_id)])
                    cars_run_line_1.remove(str(track.track_id))
                    cars_run_line_2.remove(str(track.track_id))
#                    cimg = imgcv[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
                    
                cv2.rectangle(imgcv,
                              (int(bbox[0]), int(bbox[1])),
                              (int(bbox[2]), int(bbox[3])),
                              (0,0,255),
                              lineType=2, thickness=8)
            else:
                cv2.rectangle(imgcv,
                              (int(bbox[0]), int(bbox[1])),
                              (int(bbox[2]), int(bbox[3])),
                              (255,255,255),
                              2)                           
            '''
            cv2.rectangle(imgcv,
                         (int(bbox[0]), int(bbox[1])),
                         (int(bbox[2]), int(bbox[3])),
                         (255, 255,255),
                         2)
            '''
            cv2.putText(imgcv, 
                        str(track.track_id),
                        (int(bbox[0]), int(bbox[1])),
                        0, 5e-3 * 200, (0,255,0),2)
 
        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(imgcv,
                          (int(bbox[0]), int(bbox[1])), 
                          (int(bbox[2]), int(bbox[3])),
                          (255,0,0), 
                          2)


    def detect_image(self, image, path, encoder, tracker):
        if self.is_fixed_size:
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(
                image, 
                tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
 
        #print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300
        thickness = 5

        boxs = []
        my_class = ['traffic light', 'car']
        imgcv = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            if predicted_class not in my_class:
                continue
            box = out_boxes[i]
            score = out_scores[i]  

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            img2 = imgcv[top:bottom, left:right]

            cv2.line(imgcv,(self.line[0][0],self.line[0][1]), (self.line[1][0], self.line[1][1]),(255,0,0),5)
            cv2.line(imgcv,(self.line[2][0],self.line[2][1]), (self.line[3][0], self.line[3][1]),(255,0,0),5)

            if predicted_class == 'traffic light':
                color = self.get_color(img2)
                cv2.imwrite('images/triffic/'+path+str(i) + '.jpg', img2)
                if color== 'red' or color == 'red2':
                    ####################################################
                    self.straight = False

                    cv2.rectangle(imgcv, (left, top), (right, bottom), color=(0, 0, 255),
                                  lineType=2, thickness=8)
                    cv2.putText(imgcv, '{0} {1:.2f}'.format(color, score),
                                (left, top - 15),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, (0, 0, 255), 4,
                                cv2.LINE_AA)
                    #self.car_process(imgcv)
                elif color == 'green':
                    self.straight = True

                    cv2.rectangle(imgcv, (left, top), (right, bottom), color=(0, 255, 0),
                                  lineType=2, thickness=8)
                    cv2.putText(imgcv, '{0} {1:.2f}'.format(color, score),
                                (left, top - 15),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, (0, 255, 0), 4,
                                cv2.LINE_AA)
            else:   # car
                '''
                run_red = self.intersection(self.line,top,left,bottom,right, 0) or self.intersection(self.line,top,left,bottom,right, 2)
                if self.straight == False and run_red: # red light
                    cv2.rectangle(imgcv, (left, top), (right, bottom), color=(0, 0, 255),
                                  lineType=2, thickness=8)
                    cv2.putText(imgcv, '{0} {1:.2f}'.format(predicted_class, score),
                                (left, top - 15),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, (0, 0, 255), 4,
                                cv2.LINE_AA)
                 '''
                #cimg = imgcv[top:bottom,left:right]
                #self.visual_draw_position(cimg)              
                image = Image.fromarray(imgcv)
                x = int(box[1])
                y = int(box[0])
                w = int(box[3]-box[1])
                h = int(box[2]-box[0])
                if x < 0 :
                    w = w + x
                    x = 0
                if y < 0 :
                    h = h + y
                    y = 0 
                boxs.append([x,y,w,h])
                #boxs = yolo.detect_image(image, 'pic')

            
                #cv2.imshow('', imgcv)
        self.car_tracker(boxs, imgcv, encoder, tracker)
        return imgcv
 
    def close_session(self):
        self.sess.close()
 
'''
if __name__ == '__main__':
    yolo = YOLO()
    output = 'image/output222.avi'
    video_full_path = 'image/test2.mp4'

   # 参数定义
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

   # deep_sort 目标追踪算法 
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric(
        		"cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
 
    cap = cv2.VideoCapture(video_full_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)  # 设置要获取的帧号

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
        image = yolo.detect_image(image,'pic')
        cv2.imshow('result', image)
        out.write(image)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
'''