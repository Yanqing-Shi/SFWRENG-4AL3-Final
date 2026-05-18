import pandas as pd
import numpy as np
from PIL import ImageTk, Image
from tkinter import *
from tkinter import messagebox
from time import sleep

file_dir = "./thumbnails/"


class Classifyer:
    def __init__(self, csv_file, selected_ind):
        self.ind=selected_ind
        self.load_dataset(csv_file)
        self.root = Tk()
        self.current_img = 0
        self.title = Label(self.root, text="", font=("*", 25))
        self.title.pack(side=TOP, pady=20)
        self.photo_panel = Label(self.root, height=300, width=400)
        self.photo_panel.pack(side=TOP, padx=5, fill=BOTH, expand=True)
        self.toolbar = Frame(self.root)
        self.toolbar.pack(side=BOTTOM, pady=20)
        self.prev_button = Button(self.toolbar, text="< previous", width=10)
        self.next_button = Button(self.toolbar, text="next >", width=10)
        self.prev_button.pack(side=LEFT)
        self.next_button.pack(side=RIGHT)
        self.curr_label = Label(self.toolbar, text="", font=("*", 25), width=5)
        self.curr_label.pack(anchor=CENTER, fill=X, expand=True, padx=20)
        self.refreash_page()
        self.root.bind("<Key>", self.eventcatcher)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
        
        

    def refreash_page(self):
        try:
            img = ImageTk.PhotoImage(
                file=file_dir
                + str(list(self.photo_ids.values())[self.current_img])
                + ".jpg"
            )
            self.photo_panel.config(image=img, text=None)
            self.photo_panel.image = img
            lbl = self.df.at[list(self.photo_ids.keys())[self.current_img], "aov"]
            if lbl == 1.0:
                self.curr_label.config(fg="#1e1")
                lbl_txt = "Side"
            elif lbl == 0.0:
                self.curr_label.config(fg="#e11")
                lbl_txt = "!Side"
            else:
                self.curr_label.config(fg="#aae")
                lbl_txt = "No Data"
            self.curr_label.config(text=lbl_txt)
        except:
            self.photo_panel.config(image=None, text="file error")
        self.title.config(text=f"{list(self.photo_ids.values())[self.current_img]}.jpg")

    def load_dataset(self, csv_file):
        self.df = pd.read_csv(file_dir + "catalog.csv")
        ind=self.ind
        '''
        i = self.df[((self.df.aov == 1) | (self.df.aov == 0))].index
        df = self.df.drop(i)
        '''
        df=self.df.iloc[ind]
        self.photo_ids = dict(df["id"])
        self.len = len(self.photo_ids)

    def eventcatcher(self, event):
        match event.keysym:
            case "Right":
                self.current_img += 1
            case "Left":
                self.current_img -= 1
            case "0":
                self.df.at[list(self.photo_ids.keys())[self.current_img], "aov"] = 0
                self.refreash_page()
                self.current_img += 1
                self.root.update()
                sleep(0.2)
            case "1":
                self.df.at[list(self.photo_ids.keys())[self.current_img], "aov"] = 1
                self.refreash_page()
                self.current_img += 1
                self.root.update()
                sleep(0.2)
        if self.current_img == self.len:
            self.current_img = 0
        if self.current_img == -1:
            self.current_img = self.len - 1
        self.refreash_page()

    def on_closing(self):
        resp = messagebox.askyesnocancel("Save?", "Save changes?")
        if resp:
            self.df.to_csv(file_dir + "catalog.csv")
        if resp is None:
            return
        self.root.destroy()
    



#a = Classifyer("catalog.csv")
