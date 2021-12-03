from PyQt5.QtCore import QFileSelector
import snap7
import numpy as np

class PLC(object):

    def __init__(self):
        
        # Init Variables
        self.IP = '192.168.0.1'       # IP của PLC
        self.slot = 1                   # Lấy trong TIA Portal
        self.rack = 0                   # Lấy trong TIA Portal
        self.DBNumber = 1               # Data Block cần nhận dữ liệu (DB1, DB2,...)
        self.dataStart = 1              # Vị trí bit con trỏ nhận dữ liệu
        self.dataSize = 254             # Độ dài của data (1 byte, 4 bytes, 8 bytes,...)
        self.data = np.zeros(42)        # Biến truyền data cho PLC
    
    # Test Connection with PLC
    def testConnection(self):
        plc = snap7.client.Client()
        try:
            plc.connect(self.IP, self.rack, self.slot)
        except Exception as e:
            print("Connection Error!")
        finally:
            if plc.get_connected():
                plc.disconnect()
                print("Connection Success!")

Controller = PLC()
Controller.testConnection()

plc = snap7.client.Client()
plc.connect(Controller.IP, Controller.rack, Controller.slot)

data = plc.db_read(Controller.DBNumber, 256, 1)
print(data)
snap7.util.set_bool(data, 0, 0, 1)
plc.db_write(Controller.DBNumber, 256, data)
