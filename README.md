This is a novel multimodality keystroke dynamics dataset, encompassing keystroke dynamics data for 20 distinct PINs collected from 97 users, totaling 18,935 unique entries. The dataset captures user data under two behavioral states: walking and sitting. It consists of two complementary sub-databases: a Motion Sensor Database and a Touch Database, each recording specific dimensions of user interaction. The data of touch sensors are recorded at a maximum rate of 120 Hz. The sensor sampling rate is configured to SensorManager.SENSOR_DELAY_NORMAL, where the sampling frequency is 5 Hz.

Database Structure

1. Motion Sensor Database
This sub-database records data from motion sensors, with the following columns:
Column Name	 Description
Time	         Timestamp of the sensor reading
SensorType	 Type of the motion sensor
X	         X-axis data of the motion sensor
Y	         Y-axis data of the motion sensor
Z	         Z-axis data of the motion sensor
Posture	         User's behavioral state (walk / sit)
PIN	         Type of PIN entered by the user
Sample ID	 Unique identifier for the sample
UUID	         Unique user identifier

2. Touch Database
This sub-database captures touch interaction data from the device screen, with the following columns:
Column Name	Description
ACTION_TYPE	Type of touch action, including lift, move, and press
Time	        Timestamp of the touch action
X	        X-coordinate of the touch point on the screen
Y	        Y-coordinate of the touch point on the screen
SizeMajor	Maximum diameter of the touch area
SizeMinor	Minimum diameter of the touch area
Orientation	Orientation of the finger during touch
Pressure	Pressure applied during the touch
Size	        Area of the touch region
Posture	        User's behavioral state (walk / sit)
PIN             Type of PIN entered by the user
Sample ID	Unique identifier for the sample
UUID	        Unique user identifier

3. The patterns of 20 PINs
PIN    Pattern   
194012 YYYYMM 
201412 YYYYMM
400101 YYMMDD 
141231 YYMMDD
194011 YYYYMD 
201499 YYYYMD
121940 MMYYYY 
122914 MMYYYY
010140 MMDDYY 
123114 MMDDYY
111940 MDYYYY 
992914 MDYYYY
111111 One Digit Repeated 
147258 Numpad Patterns
123456 Sequential Numbers 
585520 Chinese Elements
121212 Couplets Repeated 
112233 Double Sequential
136136 Triple Repeated 
111222 Triple Sequential
