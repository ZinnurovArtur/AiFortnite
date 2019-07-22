#from pynput.mouse import Controller
import time
import Keyboard
import  pyautogui
import win32api,win32con
time.sleep(5)
#mouse = Controller()
#print("position"+ str(mouse.position))

#mouse.position = (400,322)
#mouse.move(5,-5)

x1 = 400
y1 = 321


'''
for i in list(range(4))[:: -1]:
    print(i + 1)
    hC = win32api.LoadCursor(0, win32con.IDC_ARROW)
    print(win32api.SetCursor(hC))
    time.sleep(1)
last_time = time.time()
'''

def test(x,y):



    #win32api.mouse_event(win32con.MOUSEEVENTF_MOVE |
                         #win32con.MOUSEEVENTF_ABSOLUTE, int(1), int(3))
    Keyboard.MouseMoveTo(-63,-32)






for i in range(4)[::-1]:
    print(i+1)
    time.sleep(1)
    last_time = time.time()
test(0,0)