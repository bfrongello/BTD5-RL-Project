import pyautogui
#from Towers import tower_from_dict
from readInGameValues import get_cash, get_round

class GameEnv:
    def __init__(self):
        self.difficulty = "Easy"
        self.rows = 12
        self.columns = 16
        self.screen_width, self.screen_height = pyautogui.size()
        self.center_x, self.center_y = self.screen_width // 2, self.screen_height // 2
        self.x_offsets = [
            -900, -800, -700, -600, -500,
            -400, -300, -200, -100, 0,
            100, 200, 300, 400, 500, 600
        ]

        self.y_offsets = [
            -670, -570, -470, -370, -270,
            -170, -70, 30, 130, 230, 330, 430
        ]

        self.offsets = [
            (0, 0),
            (-900, -670),
            (600, -670),
            (600, 430),
            (-900, 430),
        ]

    def select_with_mouse(self, x_offset, y_offset):
        '''
        Selects On Screen with Mouse
        '''
        target_x = self.center_x + x_offset
        target_y = self.center_y + y_offset
        pyautogui.moveTo(target_x, target_y)
        pyautogui.sleep(1)
        pyautogui.click()

    def place_tower(self, x_offset, y_offset, tower):
            '''
            Places a Tower
            '''
            beforeCash = get_cash()
            failsafe = 0
            target_x = self.center_x + x_offset
            target_y = self.center_y + y_offset
            while True:
                pyautogui.moveTo(target_x, target_y)
                pyautogui.press(tower.get_hotkey())
                pyautogui.click()
                pyautogui.sleep(0.1)
                if beforeCash != get_cash():
                    return
                elif failsafe == 25: # In case all slots are filled and state doesn't reflect this add failsafe to break loop
                    return
                else: 
                    failsafe += 1

    def start_next_round(self):
        '''
        Starts Next Round using the "Space Bar" hotkey
        '''
        #Pressed Twice to Enable Double Speed. Only Need Once
        if get_round() == 0:
            pyautogui.press("space")
            pyautogui.press("space") 
        else:
            pyautogui.press("space")

    def reset_after_loss(self):
        '''
        Resets the Game After Losing All Lives
        '''
        pyautogui.sleep(1)
        self.select_with_mouse(-170, 65)
        pyautogui.sleep(1)

    def reset_after_win(self):
        '''
        Resets the Game After Winning
        '''
        pyautogui.sleep(1)
        pyautogui.press("escape") 
        pyautogui.sleep(1)
        pyautogui.press("space") 
        if self.difficulty == "Easy":
            x_offset, y_offset = -600, -250
        if self.difficulty == "Medium":
            x_offset, y_offset = -200, -250
        if self.difficulty == "Hard":
            x_offset, y_offset = 200, -250       
        if self.difficulty == "Impoppable":
            x_offset, y_offset = 600, -250    

        pyautogui.sleep(1)
        self.select_with_mouse(x_offset, y_offset)
        pyautogui.sleep(1)
        self.select_with_mouse(600, 200)
        pyautogui.sleep(5)

    def reset_mid_game(self):
        '''
        Resets the Game Using Mid Game Menu
        '''
        pyautogui.sleep(1)
        pyautogui.press("escape") # Open Pause Menu
        self.select_with_mouse(-300, -250 ) # Click Reset Button
        self.select_with_mouse(300, 250 ) # Click Reset Confirmation
        pyautogui.sleep(3)

    def select_upgrade(self):
        return

# Create a GameEnvironment object and call its move_mouse method
#game_env = GameEnvironment()
