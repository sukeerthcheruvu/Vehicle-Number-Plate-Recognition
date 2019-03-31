import Tkinter
import tkFileDialog
#from Tkinter import *
#from PIL import Image

import Main
mainObj=Main.Image()
def selectfile():
    global imageFile
    imageFile = tkFileDialog.askopenfile(parent=root,title='Choose a file')
    #print imageFile.name
    mainObj.main(imageFile.name)
root= Tkinter.Tk()

    
root.minsize(width=900, height=600)
root.configure(background='#33c7f7')

var = Tkinter.StringVar()
var.set("\t\t\tVEHICLE NUMBER PLATE AND LOGO IDENTIFICATION SYSTEM\t\t\t\n")
label = Tkinter.Label( root, textvariable=var, relief=Tkinter.RAISED,bd=10,pady=5)
label.config(font=("Courier", 22,"bold"))
label.pack(pady=50)
b = Tkinter.Button(root, text="Select Image",height=5, width=32,command=selectfile)
b.pack(pady=100)
var1 = Tkinter.StringVar()
var1.set("Designed By\nSukeerth Cheruvu and Anusha Katuru\n Project Guide: Dr. M. Swamy Das")
label1 = Tkinter.Label( root, textvariable=var1, relief=Tkinter.RAISED,bd=10,pady=5)
label1.config(font=("Courier", 16))
label1.pack(pady=50)
root.mainloop()
try:
    root.destroy()
except Tkinter.TclError:
    pass
    
