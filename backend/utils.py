import cv2

img = cv2.imread("image.jpg")
cv2.imshow("The Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()