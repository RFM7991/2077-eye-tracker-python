import cv2
import numpy as np
import matplotlib.pyplot as plt


def graph_pupils(captures, diameter=True, radius=False, area=False):
    if diameter:
        plt.figure()
        plt.plot([cap[2] for cap in captures], marker='o')
        plt.title('Pupil Diameter Over Time')
        plt.xlabel('Frame')
        plt.ylabel('Diameter (pixels)')
        plt.grid(True)
        plt.show(block=False)
    if radius:
        plt.figure()
        plt.plot([cap[1] for cap in captures], marker='o')
        plt.title('Pupil Radius Over Time')
        plt.xlabel('Frame')
        plt.ylabel('Radius (pixels)')
        plt.grid(True)
        plt.show(block=False)
    if area:
        plt.figure()
        plt.plot([cap[0] for cap in captures], marker='o')
        plt.title('Pupil Area Over Time')
        plt.xlabel('Frame')
        plt.ylabel('Area (pixels)')
        plt.grid(True)
        plt.show(block=False)
