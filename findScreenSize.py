import pyautogui
from pynput import mouse

def on_click(x, y, button, pressed):
    if pressed:
        screen_width, screen_height = pyautogui.size()
        center_x, center_y = screen_width / 2, screen_height / 2
        relative_x, relative_y = x - center_x, y - center_y
        print(f"Relative Position from Center: ({relative_x}, {relative_y})")
        return False  # Returning False to stop the listener

# Set up the listener
with mouse.Listener(on_click=on_click) as listener:
    listener.join()
