import ctypes
import time
import win32api
import win32con

SendInput = ctypes.windll.user32.SendInput
W = 0x11
A = 0x1E
S = 0x1F
D = 0x20


PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]
class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg",ctypes.c_ulong),
                ("wParamL",ctypes.c_short),
                ("wParamH",ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx",ctypes.c_long),
                ("dy",ctypes.c_long),
                ("mouseData",ctypes.c_ulong),
                ("dwFlags",ctypes.c_ulong),
                ("time",ctypes.c_ulong),(
            "dwExtraInfo",PUL)]
class Input_I(ctypes.Union):
    _fields_ = [("ki",KeyBdInput),("mi",MouseInput),("hi",HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),("ii",Input_I)]


def PressKey(hexKey):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0,hexKey,0x0008,0,ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1),ii_)
    ctypes.windll.user32.SendInput(1,ctypes.pointer(x),ctypes.sizeof(x))

def ReleaseKey(hexKey):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKey, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x),ctypes.sizeof(x))

def MouseMoveTo(x, y):

    width=  65536 * x /ctypes.windll.user32.GetSystemMetrics(win32con.SM_CXSMICON)+1
    height = 65536 * y/ ctypes.windll.user32.GetSystemMetrics(win32con.SM_CYSMICON)+1

    xl = ctypes.c_long()
    xl.value = int(width)
    yl = ctypes.c_long()
    yl.value = int(height)


    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.mi = MouseInput(x, y, 0, (0x0001 | 0x0008) , 0, ctypes.pointer(extra))

    command = Input(ctypes.c_ulong(0), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))






if __name__ == '__main__':
    PressKey(0x011)
    time.sleep(1)
    ReleaseKey(0x11)
    time.sleep(1)