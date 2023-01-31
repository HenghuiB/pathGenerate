import cv2
import imageio
from matplotlib import pyplot as plt, animation
import networkx as nx

Start='A1'
Target='C3'

import numpy as np


def move(p1, p2, p3, positions):
   l1 = [p2[0] - p1[0], p2[1] - p1[1]]
   l2 = [p3[0] - p2[0], p3[1] - p2[1]]
   dot = np.dot(l1, l2)
   angle = np.degrees(np.arccos(dot / (np.linalg.norm(l1) * np.linalg.norm(l2))))
   if p2[1] < p3[1]:
      angle = 360 - angle
   return angle
   # for i in range(len(positions)):
   #    positions[i] = [(positions[i][0] - p2[0]) * np.cos(angle * np.pi / 180) -
   #                    (positions[i][1] - p2[1]) * np.sin(angle * np.pi / 180),
   #                    (positions[i][0] - p2[0]) * np.sin(angle * np.pi / 180) +
   #                    (positions[i][1] - p2[1]) * np.cos(angle * np.pi / 180)]
   # return positions


def generateGraph():
   vs= []
   edges=[]
   for i in range(30):
      vs.append(['A' + str(i), (i, 0)])
      edges.append(('A'+str(i) , 'A'+ str(i+1)))
      vs.append(['B' + str(i), (i, 19)])
      edges.append(('B'+str(i) , 'B'+ str(i+1)))
   edges.pop()
   edges.pop()
   for i in range(1, 19):
      vs.append(['C' + str(i), (0, i)])
      edges.append(('C'+str(i) , 'C'+ str(i+1)))
      vs.append(['D' + str(i), (19, i)])
      edges.append(('D'+str(i) , 'D'+ str(i+1)))
      vs.append(['E' + str(i), (29, i)])
      edges.append(('E'+str(i) , 'E'+ str(i+1)))
   edges.pop()
   edges.pop()
   edges.pop()
   edges.append(('A0','C1'))
   edges.append(('A19','D1'))
   edges.append(('A29','E1'))
   edges.append(('B0','C18'))
   edges.append(('B19','D18'))
   edges.append(('B29','E18'))

   return vs,edges

plt.rcParams["figure.figsize"] = [19.2, 10.8]
plt.rcParams["figure.autolayout"] = False
fig = plt.figure()

G = nx.Graph()

names = ['C']
positions = [(0, 0)]
edges = []

tempGraph=generateGraph()
for i in tempGraph[0]:
   if i[0]==Start:
      positions[0]=i[1]
for i in tempGraph[0]:
   names.append(i[0])
   positions.append(i[1])
edges = tempGraph[1]
for index, name in enumerate(names):
    G.add_node(name, pos=positions[index])
G.add_edges_from(edges)


layout = dict((n, G.nodes[n]["pos"]) for n in G.nodes())

frm=0

nx.draw(G, pos=layout, with_labels=False, node_size=[400]+[50 for i in range(len(names)-1)])
plt.savefig('./picture/frame'+str(frm)+'.png')
frm+=1

path=nx.shortest_path(G,source = Start, target = Target)

initRec=0
RotateFrm=[]

def animate(frame):
   fig.clear()
   global initRec,positions,layout,RotateFrm,frm
   currPos=layout['C']
   if initRec<len(path):
      if abs(currPos[0]-layout[path[initRec]][0])<1e-4:
         if abs(currPos[1]-layout[path[initRec]][1])<1e-4:
            if initRec!=0 and initRec!=len(path)-1:
               RotateFrm.append([frm,move(layout[path[initRec-1]],layout[path[initRec]],layout[path[initRec + 1]],[])])
               # RotateFrm.append(move([layout[path[initRec-1]]],layout[path[initRec]],layout[path[initRec+1]],positions))
            initRec+=1
         elif currPos[1]<layout[path[initRec]][1]:
            # if (move([currPos[0], currPos[1]], [currPos[0], currPos[1] + 0.05], layout[path[initRec]], positions)-0)>1e-2:
            #    RotateFrm.append([frm,move([currPos[0], currPos[1]], [currPos[0], currPos[1] + 0.05], layout[path[initRec]], positions)])
            # positions = move([currPos[0], currPos[1]], [currPos[0], currPos[1] + 0.05],layout[path[initRec]],positions)
            # layout = dict((n, G.nodes[n]["pos"]) for n in G.nodes())
            layout['C'] = (currPos[0], currPos[1] + 0.05)
         else:
            # if (move([currPos[0], currPos[1]], [currPos[0], currPos[1] - 0.05], layout[path[initRec]], positions)-0)>1e-2:
               # RotateFrm.append([frm,move([currPos[0], currPos[1]], [currPos[0], currPos[1] - 0.05], layout[path[initRec]], positions)])
            layout['C'] = (currPos[0], currPos[1] - 0.05)
      elif abs(currPos[1]-layout[path[initRec]][1])<1e-4:
         if currPos[0]<layout[path[initRec]][0]:
            # if (move([currPos[0], currPos[1]], [currPos[0]+0.05, currPos[1]], layout[path[initRec]], positions)-0)>1e-2:
            #    RotateFrm.append([frm,move([currPos[0], currPos[1]], [currPos[0]+0.05, currPos[1]], layout[path[initRec]], positions)])
            layout['C'] = (currPos[0] + 0.05, currPos[1])
         else:
            # if (move([currPos[0], currPos[1]], [currPos[0]-0.05, currPos[1]], layout[path[initRec]], positions)-0)>1e-2:
            #    RotateFrm.append([frm,move([currPos[0], currPos[1]], [currPos[0]-0.05, currPos[1]], layout[path[initRec]], positions)])
            layout['C'] = (currPos[0] - 0.05, currPos[1])
      nx.draw(G, pos=layout,with_labels=False, node_size=[400]+[50 for i in range(len(names)-1)])
      plt.savefig('./picture/frame' + str(frm) + '.png')
      frm += 1

ani = animation.FuncAnimation(fig, animate, frames=22*len(path), interval=1, repeat=False)

plt.show()

with open('rec.txt','w') as f:
   f.write(str(RotateFrm))

def makeBorder(n):
   for i in range(n):
      img = cv2.imread('./picture/frame' + str(i) + '.png')
      img2 = cv2.copyMakeBorder(img, 300, 300, 300, 300, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255, 0))
      cv2.imwrite('./picture/frame' + str(i) + '.png', img2)

makeBorder(frm)

def clipPic(n):
   for i in range(n):
      img = cv2.imread('./picture/frame' + str(i) + '.png', 0)
      img = cv2.GaussianBlur(img, (5, 5), 2, 2)
      img_th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 3)
      circles = cv2.HoughCircles(img_th, cv2.HOUGH_GRADIENT, 2, minDist=30, param1=200, param2=40, minRadius=10,
                                 maxRadius=20)

      try:
         center = [int(circles[0][0][0]), int(circles[0][0][1])]
      except:
         continue
      image = cv2.imread('./picture/frame' + str(i) + '.png')

      image = image[center[1] - 300:center[1] + 300, center[0] - 300:center[0] + 300]

      cv2.imwrite('./picture/clip/frame' + str(i) + '.png', image)

clipPic(frm)

def rotate(n):
   l = []
   with open('rec.txt', 'r') as f:
      temp = f.read()[:-2].split('],')
      for i in temp:
         l.append([float(i[2:].split(',')[0]), float(i[2:].split(',')[1])])

   newL = []
   for i in l:
      if abs(i[1] - 0) < 1e-2:
         continue
      else:
         newL.append(i)

   for i in newL:
      for q in range(int(i[0]), n):
         try:
            image = cv2.imread('./picture/clip/frame' + str(q) + '.png')
            # if i == newL[0]:
            #    image = cv2.imread('./picture/clip/frame' + str(q) + '.png')
            # else:
            #    image = cv2.imread('./picture/clip/rotate/frame' + str(q) + '.png')
            (h, w) = image.shape[:2]
            (cX, cY) = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D((cX, cY), -int(i[1]), 1.0)
            rotated = cv2.warpAffine(image, M, (w, h))
            cv2.imwrite('./picture/clip/frame' + str(q) + '.png', rotated)
         except:
            continue

rotate(frm)

def toGif(n):
   imgs = []
   for i in range(1, n):
      try:
         imgs.append(imageio.v2.imread("./picture/clip/frame"+str(i)+".png"))
      except:
         continue
   output_name = "car.gif"
   imageio.mimsave(output_name, imgs, 'GIF', duration=0.01)

toGif(frm)