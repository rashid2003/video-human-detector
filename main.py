import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import sys

filename = sys.argv[1]
file_size = (1920,1080); scale_ratio = 1

# We want to save the output to a video file
output_filename = 'output/output.mp4'; output_frames_per_second = 20.0

def main():
  hog = cv2.HOGDescriptor()
  hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

  cap = cv2.VideoCapture(filename)
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  result = cv2.VideoWriter(output_filename,
                           fourcc,
                           output_frames_per_second,
                           file_size)

  while cap.isOpened():
    ret, frame = cap.read()
    if ret:
      width = int(frame.shape[1] * scale_ratio)
      height = int(frame.shape[0] * scale_ratio)
      frame = cv2.resize(frame, (width, height))
      orig_frame = frame.copy()

      ( bounding_boxes, weights ) = hog.detectMultiScale(frame,
                                                            winStride=(8,8),
                                                            padding=(32,32),
                                                            scale=1.05)

      # remove the boxes with confidence less than 0.5
      idx = 0
      while idx < len(weights):
          if weights[idx] < 0.8:
              weights = remove_element(weights, idx)
              bounding_boxes = remove_element(bounding_boxes, idx)
          else:
                idx += 1

      for (x, y, w, h) in bounding_boxes:
            cv2.rectangle(orig_frame,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
             2)

      bounding_boxes = np.array([[x, y, x + w, y + h] for (
                                x, y, w, h) in bounding_boxes])

      selection = non_max_suppression(bounding_boxes,
                                      probs=None,
                                      overlapThresh=0.45)

      for (x1, y1, x2, y2) in selection:
        cv2.rectangle(frame,
                     (x1, y1),
                     (x2, y2),
                     (0, 255, 0),
                      4)

      result.write(frame)


      # Display the number of people detected
      font        = cv2.FONT_HERSHEY_SIMPLEX
      top_left    = (50,60)
      font_scale  = 1
      font_color  = (255,255,255)
      thickness   = 1
      line_type   = 2

      cv2.putText(frame, 'People: ' + str(len(selection)),
        top_left,
        font,
        font_scale,
        font_color,
        thickness,
        line_type
      )

      cv2.imshow("Frame", frame)

      # q == quit
      if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    else:
      break

  cap.release()
  result.release()
  cv2.destroyAllWindows()


# helper function
def remove_element(array, index):
    return np.delete(array, index, 0)


if __name__ == '__main__':
  main()
