# importing required module
import torchaudio
import librosa
import pydub
import IPython
import pygame
import os
from tkinter import*
 
root = Tk()
root.title('GeeksforGeeks sound player')  #giving the title for our window
root.geometry("500x400")
 
pygame.mixer.init()
# making function
dir_name = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\raw2_mono_hospital\1598482996718_NA\0'
file_list = (os.path.join(dir_name, f) for f in os.listdir(dir_name))
def play():
    file = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_snoring_subset\raw2_mono_hospital\1598482996718_NA\1\1598482996718_70_101.0_102.0_010.wav'
    # y, sr = librosa.load(file)
    # IPython.display.Audio(data=y, rate=sr)

    pygame.mixer.music.load(next(file_list))
    pygame.mixer.music.play(loops=0)
 
# title on the screen you can modify it   
title=Label(root,text="Audio player",bd=6,relief=GROOVE,
            font=("times new roman",40,"bold"),bg="white",fg="blue")
title.pack(side=TOP,fill=X)
 
# making a button which trigger the function so sound can be playeed
play_button = Button(root, text="Play Song", font=("Helvetica", 32),
                     relief=GROOVE, command=play)
play_button.pack(pady=20)
 
info=Label(root,text="Click on the button above to play song ",
           font=("times new roman",10,"bold")).pack(pady=20)
root.mainloop()

# if __name__ == '__main__':
    