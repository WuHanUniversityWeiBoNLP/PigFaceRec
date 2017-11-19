import cv2

for i in range(1,31):
    #you should put 30 mp4s of pig under data dec
    file = 'data/train/{}.mp4'.format(i)
    cap = cv2.VideoCapture(file)
    c = 0
    while cap.isOpened():
        ret,frame = cap.read(0)
        #cv2.imshow('frame',frame)
        c += 1
        #resize = cv2.resize(frame, (224, 224))
        #extract 2000 pics for train,400 pics for test with every categories
        if c <= 2000:
            cv2.imwrite('data/train_data/{}/{}.jpg'.format(i,c),frame)
        elif c <= 2400:
            cv2.imwrite('data/test_data/{}/{}.jpg'.format(i,c),frame)
        else:
            break


