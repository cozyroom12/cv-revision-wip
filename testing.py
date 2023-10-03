import matplotlib.pyplot as plt
import cv2

image = cv2.imread('book_cover.jpg')

# creating circle with the mouse

def create_circle(event, x, y, flags, param):
  if event == cv2.EVENT_RBUTTONDOWN:
    cv2.circle(image, (x,y), 100, (0, 0, 255), thickness=10)


cv2.namedWindow(winname='dog')
cv2.setMouseCallback('dog', create_circle)

while True:
  cv2.imshow('dog', image)
  if cv2.waitKey(20) & 0xFF == 27:
    break

cv2.destroyAllWindows()