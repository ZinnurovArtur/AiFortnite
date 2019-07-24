import win32gui, time
import pyautogui
import win32api
import win32con
import sys

def dotrun(x2,y2):

    try:

     # while True:
        x = int(400)  # 1920 / 2 = 960 for X position on screen.
        y = int(321)  # 1200 / 2 = 600 for Y position on screen.


        x = x2
        y = y2

        green = win32api.RGB(0,255,0)
        color = int(green)  # Pixel color, 255 = Red
        hwnd = win32gui.WindowFromPoint((x, y))
        hdc = win32gui.GetDC(hwnd)

        x1, y1 = win32gui.ScreenToClient(hwnd, (x, y))
        win32gui.SetPixel(hdc, x1, y1, color)
        win32gui.SetPixel(hdc, x1 - 1, y1, color)
        win32gui.SetPixel(hdc, x1 + 1, y1, color)
        win32gui.SetPixel(hdc, x1, y1 + 1, color)
        win32gui.SetPixel(hdc, x1 - 1, y1 - 1, color)
        win32gui.SetPixel(hdc, x1 + 1, y1 + 1, color)
        win32gui.SetPixel(hdc, x1 - 1, y1 + 1, color)
        win32gui.SetPixel(hdc, x1 + 1, y1 - 1, color)

        win32gui.SetPixel(hdc, x1 - 2, y1, color)
        win32gui.SetPixel(hdc, x1 + 2, y1, color)
        win32gui.SetPixel(hdc, x1, y1 - 2, color)
        win32gui.SetPixel(hdc, x1, y1 + 2, color)
        win32gui.SetPixel(hdc, x1 - 2, y1 - 2, color)
        win32gui.SetPixel(hdc, x1 + 2, y1 + 2, color)
        win32gui.SetPixel(hdc, x1 - 2, y1 + 2, color)
        win32gui.SetPixel(hdc, x1 + 2, y1 - 2, color)
        win32gui.ReleaseDC(hwnd, hdc)
        # time.sleep(0.1)
    except KeyboardInterrupt:
        print('exit')
        sys.exit()


def test(x2,y2):
  try:

    time.sleep(5)
    win32api.SetCursorPos((x2, y2))
    while True:

      print(win32gui.GetCursorInfo())
  except KeyboardInterrupt:
    print("exit")
    sys.exit()

dotrun(383,321)
#test(400,321)