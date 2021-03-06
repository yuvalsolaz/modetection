# modetection
Pedestrian and Smartphone Motion Mode Recognition

The possibility of using mobile devices (such as smartphones) for locating a person is becoming more and more attractive for many applications. Among them are health care services, commercial applications, emergency applications and safety services.
While in outdoors, the positioning of a person by its smartphone is usually based on Global Navigation Satellite Systems (GNSS). However, in indoor environments the availability of satellite signals cannot be guaranteed and GNSS based services can be highly degraded or totally denied. 
In such situations, one of the approaches to find the position of the smartphone is known as Pedestrian Dead Reckoning (PDR). PDR relies on the smartphone low-cost sensors such as accelerometers, gyroscopes and magnetometers. In general, after a user depend calibration phase, PDR uses the accelerometers to detect the pedestrian steps and then estimate the step length. Next, the heading is obtained from the gyroscopes and/or magnetometer. Given the pedestrian initial conditions and by using the current heading and step length size, the current pedestrian position can be found.
Recent papers showed that by identifying the mode of the smartphone (handheld, in a pocket, texting and etc) and/or the pedestrian (walking, running, elevator and etc) the accuracy of PDR algorithms can greatly be improved. In this research, we employ machine learning algorithms in order to recognize and classify the mode of the pedestrian and smartphone. Given the pedestrian and smartphone modes, appropriate algorithms are used for the PDR process. Results show that the mode recognition improves the accuracy of PDR algorithms.



Classifications categories :

            device mode classifier categories  : 
                        1. swinging in hand   
                        2. in pocket  
                        3. texting
                        4. unknown 

            pedestrian modes classifier : 
                        1. fast walking 
                        2. calm walking 
                        3. stairs
                        4. static 

data collection : 

            recording tool : android  "Physics Toolbox Suite"  application 

            sample rate : faster ( sensor maximum frequency )  == 50 Hertz
            
            sensors :
                        1. g-force - x,y,z acceleration in g units 
                        2. gyro -              wx wy wz red/sec 
                        3. magnetometer - magnetic field x y z 
                        4. barometer - if exists on the device 
                        5. light meter - just in case 


            optional : 
                        add user events logging 
                        
                        
            recording layout : 
                        single mode sessions of 30 seconds 
                        start each session with static 2 seconds  for sensor calibration 
                        go only in straight line
                        
            recurrence : 
                        keep the same mode pattern : 
                                    pocket mode use back pocket, screen toward body 
                                    text mode use the same device orientation 

pre processing  : 
           
                         1. noise reduction - high pass filter 
                         2. remove dc                      
                         3. normalize

Optional Features : 
                         1. average norm on N samples sliding window      ( N = 128) 
                         2. variance on N samples  sliding window   
                        
Notes : 
                        1. Jeans or sports pants ?
                        2. non sterilized walking environment  

