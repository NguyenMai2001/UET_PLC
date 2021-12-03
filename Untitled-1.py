# import Send_data_func as f
import codecs
import serial
import time
import wheel

time.sleep(1)
arduino = serial.Serial(port='COM3', baudrate=9600, timeout=1)
#print(user)
# while True:
# print(user)
# if user == "a":
user = "a"
if user == "a":
    print("begin")
    arduino.write(bytes(user,'utf-8'))
    time.sleep(5)
    data = arduino.readline()
    #Change byte to string
    errCheck_str = data.decode(encoding="utf-8")
    #print(len(errCheck_str))
    errCheck = []
    if(len(errCheck_str)==3):
        index = 0
    else: 
        if(len(errCheck_str)==9):
            errCheck.append("0")
            for i in range(len(errCheck_str)):
                errCheck.append(errCheck_str[i])
        else:
            for i in range(len(errCheck_str)):
                errCheck.append(errCheck_str[i])

        print("Err_Check: ",errCheck)
        if errCheck[1] == "1": index = 2
        if errCheck[7] == "1": index = 8
        if errCheck[2] == "1": index = 4
        if errCheck[4] == "1": index = 5
        if errCheck[6] == "1": index = 6
        if errCheck[5] == "1": index = 7
        if errCheck[3] == "1": index = 3
        if errCheck[0] == "1": index = 1
    dispMap =["OK" , "I2C" , "DF_S" ,"AF_D" ,"FIXD" ,"EMPT" ,"DATA" ,"AWB" ," CRC"]

    Error = dispMap[index]
    print("Error:",Error)