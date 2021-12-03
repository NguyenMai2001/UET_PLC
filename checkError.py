import codecs
import serial
import time

def err_Check():
    # Importing Libraries

    #Connect with Arduino
    arduino = serial.Serial(port='COM3', baudrate=9600, timeout=1)
    while True:
        user = input() # Taking input from user
        arduino.write(bytes(user,'utf-8'))
        time.sleep(0.05)
        value = arduino.readline()
        
        #Change byte to string
        errCheck_str = value.decode(encoding="utf-8")
        #dispMap
        dispMap =["  OK" , " I2C" , "ER_A" ,"ER_B" ,"ER_C" ,"ER_D" ,"ER_E" ,"ER_F" ," CRC"]
        #print error
        if errCheck_str[1] == "1": index = 2
        elif errCheck_str[2] == "1": index = 3
        elif errCheck_str[3] == "1": index = 4
        elif errCheck_str[4] == "1": index = 5
        elif errCheck_str[5] == "1": index = 6
        elif errCheck_str[6] == "1": index = 7
        elif errCheck_str[7] == "1": index = 8
        elif errCheck_str[0] == "1": index = 1
        else: index = 0
        Error = dispMap[index]
    return Error
    


  



