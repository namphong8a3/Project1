import pygame
import numpy as np
from datetime import datetime
gamma = 0.9
alpha = 0.5

grey=(120, 120, 120)
white=(255, 255, 255)
green=(81, 207, 152)
blue=(67, 172, 217)
black=(0, 0, 0)
red=(184, 7, 25)
green2=(67, 168, 130)
orange=(222, 92, 27)

running = True
mode=0
done=False
route=[]
n=int(input())
Edge=[]
start_location=end_location=0
bs=be=False
a=[[0]*n*n]*n*n
rewards=np.array(a)

def getcenter(location):  #Get center of the square
    x=location%n
    y=int(location/n)
    res=(x*size+int(size/2),y*size+int(size/2))
    return res

def check(i,j):
    if i>=0 and i<n and j>=0 and j<n:
        return True
    else:
        return False
#----------------------Reward-----------------------------
for i in range(0,n*n):
    a=i/n
    b=i%n
    if (check(a,b+1)):
        rewards[i][i + 1] = 1
    if (check(a,b-1)):
        rewards[i][i - 1] = 1
    if (check(a+1,b)):
        rewards[i][i + n] = 1
    if (check(a-1,b)):
        rewards[i][i - n] = 1

def check_mode(x,y):
    if x<=850 and x>=800 and y<=450 and y>=400:
        return 4
    if x>=700 and x<=900:
        if y>=100 and y<=180:
            return 1
        if y>=200 and y<=280:
            return 2
        if y>=300 and y<=380:
            return 3
        return 0
    else:
        return 0

#---------------------------Main Code--------------------------------------
def get_optimal_route(start_location, end_location):

    rewards_new = np.copy(rewards)

    ending_state = end_location

    rewards_new[ending_state, ending_state] = 999
    Q = np.array(np.zeros([n*n, n*n]))

    for i in range(100000):

        current_state = np.random.randint(0,n*n)

        playable_actions = []
        while len(playable_actions)==0:
            current_state = np.random.randint(0, n * n)
            for j in range(n*n):
                if rewards_new[current_state, j] > 0:
                    playable_actions.append(j)

        next_state = np.random.choice(playable_actions)


        TD = rewards_new[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state]
        Q[current_state, next_state] += alpha * TD

    route = [start_location]

    next_location = start_location
    print(end_location)
    while (next_location != end_location):
        starting_state = start_location
        next_state = np.argmax(Q[starting_state,])
        next_location=next_state
        route.append(next_state)
        start_location = next_location
    return route
#-----------------------------Creating environment-------------------------------------------

pygame.init()
screen=pygame.display.set_mode((1000,600))

while running:
    size=min(100,600/n)

    screen.fill(white)
    for i in range(0,n+1):
        pygame.draw.line(screen, black, (0,size*i),(n*size,size*i),2)
    for i in range(0,n+1):
        pygame.draw.line(screen, black, (size*i, 0), (size*i,n*size),2)

    pygame.draw.rect(screen, green, (700,100,200,80))
    pygame.draw.rect(screen, blue, (700,200,200,80))
    pygame.draw.rect(screen, red, (700,300,200,80))
    pygame.draw.rect(screen, orange , (800,400,50,50))

    if bs==True:
        pygame.draw.rect(screen, green, (int(start_location%n)*size,int(start_location/n)*size,size,size))
    if be == True:
        pygame.draw.rect(screen, blue, (int(end_location % n) * size, int(end_location / n) * size, size, size))

    pygame.draw.line(screen, black, (0,0), (n*size,0) , 4)
    pygame.draw.line(screen, black, (0,0), (0,n*size) , 4)
    pygame.draw.line(screen, black, (n*size,0), (n*size,n*size) , 4)
    pygame.draw.line(screen, black, (0,n*size), (n*size,n*size) , 4)
    for i in Edge:
        if i[0]==0:
            pygame.draw.line(screen, red, (i[2]*size,i[1]*size), ((i[2]+1)*size,i[1]*size), 4)
        else:
            pygame.draw.line(screen, red, (i[1]*size,i[2]*size), ((i[1])*size,(i[2]+1)*size), 4)
#---------------------------------------------------------------------------------------------------------

    if mode==4 and done==False:
        route=get_optimal_route(start_location,end_location)
        done=True
        print(route)

    if done:
        for i in range(1, len(route)):
            a=route[i-1]
            b=route[i]
            pygame.draw.line(screen, orange, getcenter(a), getcenter(b),8)
    for event in pygame.event.get():
        if(event.type==pygame.QUIT):
            running =False

        if event.type==pygame.MOUSEBUTTONDOWN:
            x,y=pygame.mouse.get_pos()
            if event.button==1:
                k=check_mode(x,y)
                if(k):
                    mode=k
                    continue
                if mode==1:
                    if(x>n*size or y>n*size or x<0 or y<0):
                        continue
                    location=int(x/size)+int(y/size)*n
                    if(location==start_location):
                        bs=False
                        start_location=None
                    else:
                        bs=True
                        start_location=location
                if mode == 2:
                    if(x>n*size or y>n*size or x<0 or y<0):
                        continue
                    location = int(x / size)  + int(y / size) * n
                    if (location == end_location):
                        be = False
                        end_location=None
                    else:
                        be = True
                        end_location = location
                if mode == 3:
                    if(x>n*size or y>n*size or x<0 or y<0):
                        continue
                    a=round(x/size)
                    b=round(y/size)
                    if(abs(a*size-x)<abs(b*size-y)):
                        h=int(y/size)
                        location=n*h+a
                        if(location):
                            rewards[location][location-1]=0
                            rewards[location-1][location]=0
                        Edge.append((1,a,h))
                    else:
                        h=int(x/size)
                        location=b*n+h
                        if(location>=n):
                            rewards[location-n][location]=0
                            rewards[location][location-n]=0
                        Edge.append((0,b,h))

    pygame.display.flip()
pygame.quit()