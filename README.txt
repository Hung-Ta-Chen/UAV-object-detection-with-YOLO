##Model training
    ##Preparation for training data
    Put your images and labels for training into PyTorch-YOLOv3/data/custom/data/images and PyTorch-YOLOv3/data/custom/data /labels respectively. Then run the following code to write file names of training images to train.txt and valid.txt.
    
        """
        import os
    
        i = 1
        train_file = open("./data/custom/train.txt", mode = 'w')
        valid_file = open("./data/custom/valid.txt", mode = 'w')
        
        for image_file in os.listdir("./data/custom/images"): 
          if i % 4 == 0:
            try:
              valid_file.write("./data/custom/images/" + image_file + "\n")
            except Exception as e:
              print(type(e))
              print(str(e))
          else:
            try:
              train_file.write("./data/custom/images/" + image_file + "\n")
            except Exception as e:
              print(type(e))
              print(str(e))
            
          i = i + 1
        
        train_file.close()
        valid_file.close()
        """
    
    ##Start training
    Assume that all requirments are met, run the following command. 
        $ python3 train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data 
        
    If you want to load specific weight and specify number of epoch, add "pretrained_weights" and "epochs" argument.
        $ python3 train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data --epochs 100 --pretrained_weights checkpoints/yolov3_ckpt_0.pth
    
    
##Uav control
    ##enter SDK mode
        $ python example.py
            -command
            -streamon
            if 'ok' Control+C end SDK mode
    
    ##change the target name in detect_uav.py
    
    ##start detecting and flying:
        $ python3 detect_uav.py --model_def config/yolov3-custom.cfg --weights_path checkpoints/yolov3_ckpt_167.pth --class_path data/custom/classes.names 
    
    ##taking screenshots:
        press space to take and save screenshots anytime
        the uav also saves a frame when reaching desired position
