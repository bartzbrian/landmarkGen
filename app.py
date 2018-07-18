import Tkinter
from Tkinter import Tk, BOTH
from ttk import Frame, Button, Style
from tkFileDialog import askdirectory
import tkMessageBox
import dlib
import glob
import os
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import PIL
import cv2
import numpy as np
import random
import sys


class Example(Frame):

    def __init__(self, parent):

        Frame.__init__(self, parent)   
        
        self.src = "" 
        self.dest = ""
        self.predictor_path = "./shape_predictor_68_face_landmarks.dat"
        
        self.parent = parent
        
        self.initUI()
        
    def rect_contains(self, rect, point):

        if point[0] < rect[0] :
            return False
        elif point[1] < rect[1] :
            return False
        elif point[0] > rect[2] :
            return False
        elif point[1] > rect[3] :
            return False
        return True
 
    # Draw a point
    def draw_point(self, img, p, color ):
        cv2.circle( img, p, 2, color, -1, cv2.LINE_AA, 0 )
     
     
    # Draw delaunay triangles
    def draw_delaunay(self, img, subdiv, delaunay_color ):
     
        triangleList = subdiv.getTriangleList();
        size = img.shape
        r = (0, 0, size[1], size[0])
     
        for t in triangleList :
             
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
             
            if self.rect_contains(r, pt1) and self.rect_contains(r, pt2) and self.rect_contains(r, pt3):
             
                cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
                cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
                cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)

 
    def delaunay(self, path, txt_path):
        try:
            txt_path = txt_path.replace(' ', '')[:-4].upper()
            new_background = (255, 255, 255)
            overlay = Image.open(path)
            mode = overlay.mode
            width, height = overlay.size        
            just_cage = Image.new(mode,(width,height),new_background)
            just_cage.save(txt_path + "_just_cage.jpg", 'jpeg')

            points_color = (0, 0, 255)
            img1 = cv2.imread(path); 
            img2 = cv2.imread(txt_path + "_just_cage.jpg")

            img1_orig = img1.copy(); 
            img2_orig = img2.copy();     
        
            size = img1.shape
            rect = (0, 0, size[1], size[0])     
            subdiv = cv2.Subdiv2D(rect); 
            points = [];     
            with open(txt_path + ".txt") as file :
                for line in file :
                    x, y = line.split()
                    points.append((int(x), int(y))) 
            for p in points :
                subdiv.insert(p)
            self.draw_delaunay( img1, subdiv, (0, 0, 0) );
            self.draw_delaunay( img2, subdiv, (0, 0, 0) ); 
     
            for p in points :
                self.draw_point(img1, p, (0,0,255))
                self.draw_point(img2, p, (0,0,255)) 
     
            cv2.imwrite(txt_path + 'cage_overlay.jpg',img1)
            cv2.imwrite(txt_path + '_just_cage.jpg',img2)
        except Exception,e:
            return

    def create_images(self, path, txt_path):
        try:
            new_background = (255, 255, 255)
            overlay = Image.open(path)
            mode = overlay.mode
            width, height = overlay.size
            radius = height/250
            just_dots = Image.new(mode,(width,height),new_background)
            draw1 = ImageDraw.Draw(overlay)
            draw2 = ImageDraw.Draw(just_dots)

            lms = open(txt_path, "r")
            lms = lms.readlines()

            for x in xrange(0,68):
                cs = lms[x].split(' ')
                draw1.ellipse((float(cs[0])-radius, float(cs[1])-radius, float(cs[0])+radius, float(cs[1])+radius), fill=(255,0,0,0))
                draw2.ellipse((float(cs[0])-radius, float(cs[1])-radius, float(cs[0])+radius, float(cs[1])+radius), fill=(0,0,0,0))

            txt_path = txt_path.replace(' ', '')[:-4].upper()
            overlay.save((txt_path + "_overlay.jpg"),'jpeg')
            just_dots.save((txt_path + "_just_dots.jpg"),'jpeg')
        except Exception,e:
            filename = os.path.basename(path)
            msg = "while trying to create the images from the image:\n\n" + filename + "\n\nthe app encountered the following error:\n\n" + str(e) + "\n\n Please make sure there are no spaces in your file name or the names of either your source or destination folder, then try again"
            self.message(msg)




    def generate_landmarks(self):
        import skimage
        import skimage.io

        if self.src == "":
            self.message("please pick a source folder first")
        elif self.dest == "":
            self.message("please pick a destination folder first")
        else:
            self.message("attempting to generate landmarks")
            faces_folder_path = self.src
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(self.predictor_path)



            for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
                filename = os.path.basename(f)
                try:
                    img = skimage.io.imread(f)
                except Exception,e:
                    self.message(e)
                dets = detector(img, 1)
                err = True
                for k, d in enumerate(dets):
                    shape = predictor(img, d)
                    orig_path = self.src + "/" + filename
                    os.mkdir(self.dest + "/" + filename + "/")
                    name = self.dest + "/" + filename + "/" + filename +".txt"
                    file = open(name,"w")
                    for x in xrange(0,68):
                        err = False
                        file.write(str(shape.part(x).x) + " " + str(shape.part(x).y) + "\n")
                    file.close()
                    self.create_images(orig_path,name)
                    self.delaunay(orig_path,name)

                if err:    
                    msg = "an error occurred while creating the facial landmarks for the image named:\n\n" + filename +  "\n\nTry again, and if it doesn't work again, then try a different image."
                    self.message(msg)



            for f in glob.glob(os.path.join(faces_folder_path, "*.JPG")):
                filename = os.path.basename(f)
                try:
                    img = skimage.io.imread(f)
                except Exception,e:
                    self.message(e)
                dets = detector(img, 1)
                err = True
                for k, d in enumerate(dets):
                    shape = predictor(img, d)
                    orig_path = self.src + "/" + filename
                    os.mkdir(self.dest + "/" + filename + "/")
                    name = self.dest + "/" + filename + "/" + filename +".txt"
                    file = open(name,"w")
                    for x in xrange(0,68):
                        err = False
                        file.write(str(shape.part(x).x) + " " + str(shape.part(x).y) + "\n")
                    file.close()
                    self.create_images(orig_path,name)
                    self.delaunay(orig_path,name)

                if err:    
                    msg = "an error occurred while creating the facial landmarks for the image named:\n\n" + filename +  "\n\nTry again, and if it doesn't work again, then try a different image."
                    self.message(msg)




            for f in glob.glob(os.path.join(faces_folder_path, "*.jpeg")):
                filename = os.path.basename(f)
                try:
                    img = skimage.io.imread(f)
                except Exception,e:
                    self.message(e)
                dets = detector(img, 1)
                err = True
                for k, d in enumerate(dets):
                    shape = predictor(img, d)
                    orig_path = self.src + "/" + filename
                    os.mkdir(self.dest + "/" + filename + "/")
                    name = self.dest + "/" + filename + "/" + filename +".txt"
                    file = open(name,"w")
                    for x in xrange(0,68):
                        err = False
                        file.write(str(shape.part(x).x) + " " + str(shape.part(x).y) + "\n")
                    file.close()
                    self.create_images(orig_path,name)
                    self.delaunay(orig_path,name)

                if err:    
                    msg = "an error occurred while creating the facial landmarks for the image named:\n\n" + filename +  "\n\nTry again, and if it doesn't work again, then try a different image."
                    self.message(msg)

            for f in glob.glob(os.path.join(faces_folder_path, "*.JPEG")):
                filename = os.path.basename(f)
                try:
                    img = skimage.io.imread(f)
                except Exception,e:
                    self.message(e)
                dets = detector(img, 1)
                err = True
                for k, d in enumerate(dets):
                    shape = predictor(img, d)
                    orig_path = self.src + "/" + filename
                    os.mkdir(self.dest + "/" + filename + "/")
                    name = self.dest + "/" + filename + "/" + filename +".txt"
                    file = open(name,"w")
                    for x in xrange(0,68):
                        err = False
                        file.write(str(shape.part(x).x) + " " + str(shape.part(x).y) + "\n")
                    file.close()
                    self.create_images(orig_path,name)
                    self.delaunay(orig_path,name)

                if err:    
                    msg = "an error occurred while creating the facial landmarks for the image named:\n\n" + filename +  "\n\nTry again, and if it doesn't work again, then try a different image."
                    self.message(msg)

                    
            self.message("all done! Check your destination folder for your files")
                
    def help(self):
        self.message("Hello! This program will generate the following from a JPEG image of a human face:\n\n-A text file of the pixel coordinates of 68 facial landmark features (each line of the file is an xy coordinate)\n-A JPEG image of the original face image with those landmark points overlaid\n-A JPEG image of those landmark points drawn onto a blank background\n-A JPEG image of the 'face cage' derived from those landmark points overlaid onto the original image\n-A JPEG image of the 'face cage' derived from those landmark points on a blank background\n\nThe landmark points are determined with a library called dlib, which uses machine learning algorithms to detect facial features. The cages are generated using a library called opencv, which uses an algorithm called Delaunay triangulation to draw cages from the points.\n\nThis program will only work on JPEG images. The file extensions of your images must be one of the following: .jpg, .jpeg, .JPG, or .JPEG. Your file names and folder names CANNOT contain spaces, or the program won't work. None of your images should contain multiple faces.\n\nTo use the program, click the 'select images folder' button, and select the folder that contains your face images. Then click the 'select destination folder' button, and pick a folder where you want the results to be output. The output folder should be empty, or the program may not work. Finally, click the 'generate landmarks' button, and your files should be created. This may take some time, depending on how many images there are, and how large each image is. You may encounter error messages if something goes wrong, but hopefully that won't happen!")

    def message(self, text):
        tkMessageBox.showinfo("Message", text)

    def ask_src(self):
        src_path = askdirectory()
        self.src = src_path 
        print self.src  

    def ask_dest(self):
        dest_path = askdirectory()
        self.dest = dest_path    
        
    def initUI(self):
      
        self.parent.title("Facial Landmark Generator")
        self.style = Style()
        self.style.theme_use("classic")

        self.pack(fill=BOTH, expand=1)

        src_button = Button(self, text="Select Images Folder",
            command= self.ask_src)
        src_button.place(x=50, y=50)

        dest_button = Button(self, text="Select Destination Folder",
            command= self.ask_dest)
        dest_button.place(x=300, y=50)

        exec_button = Button(self, text="Generate Landmarks",
            command= self.generate_landmarks)
        exec_button.place(x=177, y=200)

        help_button = Button(self, text="Help",
            command= self.help)
        help_button.place(x=225, y=125)

    

def main():
  
    root = Tk()
    w = 540 
    h = 310 

    # get screen width and height
    ws = root.winfo_screenwidth() # width of the screen
    hs = root.winfo_screenheight() # height of the screen

    # calculate x and y coordinates for the Tk root window
    x = (ws/2) - (w/2)
    y = (hs/2) - (h/2)

    # set the dimensions of the screen 
    # and where it is placed
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))
    app = Example(root)
    root.mainloop()  


if __name__ == '__main__':
    main() 