import cv2
import os
import argparse

def capture(src_path):
	if (not os.path.exists(src_path)):
		return 
	cap = cv2.VideoCapture(src_path)

	count = 0
	frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
	fps = cap.get(cv2.CAP_PROP_FPS)
	#print("fps: %d\nframe_num: %d\nduration: %f" % fps, frame_num, frame_num/fps)
	print("frame_num: %d" % frame_num)
	print("fps: %d" % fps)

	save_dir = "../image_240/"
	
	if (not os.path.exists(save_dir)):
	    os.mkdir(save_dir)
	 
	while(cap.isOpened()):
		ret, image = cap.read()
		
		if (not ret):
			break

		if (count % fps == 0):
			file_name = "frame%04d.jpg" % count
			file_name = os.path.join(save_dir, file_name)
			
			cv2.imwrite(file_name, image)
			
			print('Saved %s' % file_name)
		count += 1
	
	cap.release()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--dir")
	#parser.add_argument("-s", "--save")

	args = parser.parse_args()
	capture(args.dir)
