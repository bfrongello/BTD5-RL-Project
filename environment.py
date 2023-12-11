import pyautogui
import random
import torch
import torch.nn as nn
import torch.optim as optim
from Towers import tower_from_dict
import numpy as np
from readInGameValues import get_cash, get_lives, get_round
import json
import time

# Read the JSON file and convert it into Tower object
with open('Towers.json', 'r') as f:
    data = json.load(f)

towers = [
    tower_from_dict(data["DartMonkey"]),
    tower_from_dict(data["TackShooter"]),
    tower_from_dict(data["SniperMonkey"]),
    tower_from_dict(data["BoomerangMonkey"]),
    tower_from_dict(data["NinjaMonkey"]),
    tower_from_dict(data["BombTower"]),
    #tower_from_dict(data["IceTower"]),
    #tower_from_dict(data["GlueGunner"]),
    #tower_from_dict(data["MonkeyBuccaneer"]),
    #tower_from_dict(data["MonkeyAce"]),
    tower_from_dict(data["SuperMonkey"]),
    #tower_from_dict(data["MonkeyApprentice"]),
    #tower_from_dict(data["MonkeyVillage"]),
    #tower_from_dict(data["BananaFarm"]),
    #tower_from_dict(data["MortarTower"]),
    #tower_from_dict(data["DartlingGun"]),
    #tower_from_dict(data["SpikeFactory"]),
    #tower_from_dict(data["MonkeySub"]),
    #tower_from_dict(data["MonkeyEngineer"]),
    #tower_from_dict(data["BloonChipper"]),
    #tower_from_dict(data["HeliPilot"]),
]

class GameEnv:
    def __init__(self):
        self.difficulty = "Easy"
        self.screen_width, self.screen_height = pyautogui.size()
        self.center_x, self.center_y = self.screen_width // 2, self.screen_height // 2
        self.x_pix = np.linspace(-900, 700, 25)
        self.col_num = len(self.x_pix)
        self.y_pix = np.linspace(-700, 470, 25)
        self.row_num = len(self.y_pix)
        self.features_per_cell = 3 # For Tower, Upgrade Path 1, and Upgrade Path 2
        self.live_count = 200.0
        self.reset_state()


    def step(self, action):
        real_action = self.decode_action(action)
        if real_action[0] == 'place_tower':
            x_ind, y_ind, tower_type = real_action[1]
            reward = self.place_tower(x_ind, y_ind, towers[tower_type])

        elif real_action[0] == 'upgrade_tower' and action < (self.get_total_action_size() - 1 - (self.grid.size(0) * self.grid.size(1))):
            x_ind, y_ind = real_action[1]
            reward = self.upgrade_tower_1(x_ind, y_ind)
        elif real_action[0] == 'upgrade_tower' and action < (self.get_total_action_size() - 1):
            x_ind, y_ind = real_action[1]
            reward = self.upgrade_tower_2(x_ind, y_ind)
        else:
            reward = self.start_next_round()

        self.update_state()
        return self.state, reward
        
    def decode_action(self, action):
        total_actions = self.get_total_action_size() - 1 # Subtract 1 b/c Index from 0
        num_tower_types = len(towers)
        if action < self.grid.size(0) * self.grid.size(1) * num_tower_types:
            x_coord = (action // num_tower_types) % self.grid.size(0)
            y_coord = action // (num_tower_types *self.grid.size(0))
            tower_type = action % num_tower_types
            real_action = ('place_tower', (x_coord, y_coord, tower_type))
            print(f'Place Tower: Action: {action}, X:{x_coord}, Y:{y_coord}, Type: {tower_type}')
        elif action < total_actions:
            x_coord = ((action-self.grid.size(0) * self.grid.size(1) * num_tower_types) % self.grid.size(1)) 
            y_coord = ((action-self.grid.size(0) * self.grid.size(1) * num_tower_types) // self.grid.size(0)) 
            if y_coord >= self.grid.size(1):
                y_coord = ((action-self.grid.size(0) * self.grid.size(1) * num_tower_types - self.grid.size(0) * self.grid.size(1)) // (self.grid.size(0))) 
            print(f'Upgrade Tower: Action: {action}, X:{x_coord}, Y:{y_coord}')
            real_action = ('upgrade_tower', (x_coord, y_coord))
        else:
            real_action = ('start_next_round',)

        return real_action
    
    def decode_real_action(self, real_action):
        num_tower_types = len(towers)
        if real_action[0] == 'place_tower':
            x_coord, y_coord, tower_type = real_action[1]
            action = (x_coord + y_coord * self.grid.size(0)) * num_tower_types + tower_type

        elif real_action[0] == 'upgrade_tower':
            x_coord, y_coord = real_action[1]
            start_index = self.grid.size(0) * self.grid.size(1) * num_tower_types
            action = start_index + (x_coord + y_coord * self.grid.size(0))

        else:
            action = self.get_total_action_size() - 1 # Subtract 1 b/c Index from 0

        return action
    
    def select_with_mouse(self, x_offset, y_offset):
        '''
        Selects On Screen with Mouse
        '''
        target_x = self.center_x + x_offset
        target_y = self.center_y + y_offset
        pyautogui.moveTo(target_x, target_y)
        pyautogui.sleep(0.01)
        pyautogui.click()
        pyautogui.sleep(0.01)

    def place_tower(self, x_ind, y_ind, tower):
        '''
        Places a Tower
        '''
        beforeCash = get_cash()
        failsafe = 0
        target_x = self.center_x + self.x_pix[x_ind]
        target_y = self.center_y + self.y_pix[y_ind]
        if get_lives() > 0:
            while True:
                self.select_with_mouse(self.center_x - 1000, self.center_y)
                pyautogui.moveTo(target_x, target_y)
                pyautogui.sleep(0.01)
                if beforeCash != get_cash():
                    pyautogui.sleep(2)
                    beforeCash = get_cash()
                pyautogui.press(tower.get_hotkey())
                pyautogui.sleep(0.01)
                pyautogui.click()
                pyautogui.sleep(0.01)
                if beforeCash != get_cash():
                    self.grid[x_ind, y_ind, 0] = tower.get_number()
                    return 2
                elif failsafe == 1: # In case all slots are filled and state doesn't reflect this add failsafe to break loop
                    if self.grid[x_ind, y_ind, 0] == 0:
                        self.grid[x_ind, y_ind, 0] = -1 #Mark as Obfuscated
                        pyautogui.moveTo(self.center_x - 1000, self.center_y) # Move mouse away from tower to clear failed placement
                        return -1
                    pyautogui.moveTo(self.center_x - 1000, self.center_y) # Move mouse away from tower to clear failed placement
                else: 
                    failsafe += 1
        return 0
        
        

    def upgrade_tower_1(self, x_ind, y_ind):
        beforeCash = get_cash()
        failsafe = 0
        target_x = self.center_x + self.x_pix[x_ind]
        target_y = self.center_y + self.y_pix[y_ind]
        if get_lives() > 0:
            while True:
                self.select_with_mouse(self.center_x - 1000, self.center_y)
                pyautogui.sleep(0.01)
                pyautogui.moveTo(target_x, target_y)
                pyautogui.sleep(0.01)
                pyautogui.click()
                pyautogui.sleep(0.01)
                pyautogui.press(',')
                pyautogui.sleep(0.01)
                if beforeCash != get_cash():
                    self.grid[x_ind, y_ind, 1] += 1
                    return 5
                elif failsafe == 3: # In case all slots are filled and state doesn't reflect this add failsafe to break loop
                    if self.grid[x_ind, y_ind, 0] == 0:
                        self.grid[x_ind, y_ind, 0] = -1 #Mark as Obfuscated
                    return -1
                else: 
                    failsafe += 1
        return 0

    def upgrade_tower_2(self, x_ind, y_ind):
        beforeCash = get_cash()
        failsafe = 0
        target_x = self.center_x + self.x_pix[x_ind]
        target_y = self.center_y + self.y_pix[y_ind]
        if get_lives() > 0:
            while True:
                self.select_with_mouse(self.center_x - 1000, self.center_y)
                pyautogui.sleep(0.01)
                pyautogui.moveTo(target_x, target_y)
                pyautogui.sleep(0.01)
                pyautogui.click()
                pyautogui.sleep(0.01)
                pyautogui.press('.')
                pyautogui.sleep(0.01)
                if beforeCash != get_cash():
                    self.grid[x_ind, y_ind, 2] += 1
                    return 5
                elif failsafe == 3: # In case all slots are filled and state doesn't reflect this add failsafe to break loop
                    if self.grid[x_ind, y_ind, 0] == 0:
                        self.grid[x_ind, y_ind, 0] = -1 #Mark as Obfuscated
                    return -1
                else: 
                    failsafe += 1        
        return 0
        
   
    def generate_action_mask(self):
        action_mask =  torch.zeros(self.get_total_action_size()).cuda()
        for tower in towers:
            # Can I Afford to Place Tower?
            if tower.can_afford_placement(get_cash()):
                # Is there a tower placed?
                for y in range(self.grid.size(1)):
                    for x in range(self.grid.size(0)):
                        if self.grid[x, y, 0] == 0:
                            real_action = ('place_tower', (x, y, towers.index(tower)))
                            action_value = self.decode_real_action(real_action)
                            action_mask[action_value] = 1
            # Can I afford to upgrade tower?
            for y in range(self.grid.size(1)):
                for x in range(self.grid.size(0)):
                    if self.grid[x, y, 0] == tower.get_number():
                        real_action = ('upgrade_tower', (x, y))
                        action_value = self.decode_real_action(real_action)
                        if self.grid[x, y, 1] < 4 and self.grid[x, y, 2] < 2:
                            if get_cash() > tower.upgradeTableEasy[0][int(self.grid[x, y, 1])]:
                                action_mask[action_value] = 1
                            if get_cash() > tower.upgradeTableEasy[1][int(self.grid[x, y, 2])]:
                                action_mask[action_value+self.grid.size(0) * self.grid.size(1)] = 1

                        elif self.grid[x, y, 1] < 2 and self.grid[x, y, 2] < 4:
                            if get_cash() > tower.upgradeTableEasy[0][int(self.grid[x, y, 1])]:
                                action_mask[action_value] = 1
                            if get_cash() > tower.upgradeTableEasy[1][int(self.grid[x, y, 2])]:
                                action_mask[action_value+self.grid.size(0) * self.grid.size(1)] = 1
                        
                        elif self.grid[x, y, 1] < 4 and self.grid[x, y, 2] == 2:
                            if get_cash() > tower.upgradeTableEasy[0][int(self.grid[x, y, 1])]:
                                action_mask[action_value] = 1

                        elif self.grid[x, y, 1] == 2 and self.grid[x, y, 2] < 4:
                            if get_cash() > tower.upgradeTableEasy[1][int(self.grid[x, y, 2])]:
                                action_mask[action_value+self.grid.size(0) * self.grid.size(1)] = 1
            # Can I afford to upgrade tower?
            #up1, up2 = tower.can_afford_upgrade(get_cash())
            #if up1:
            #    for x in range(self.grid.size(0)):
            #        for y in range(self.grid.size(1)):
            #            if self.grid[x, y, 2] > 0:
            #if up2:
            # Else yes

        action_mask[-1] = 1 # Always allow starting the next round
        return action_mask

    def get_state(self):
        return self.state
    
    def get_state_size(self):
        return self.state.size(0)
    
    def update_state(self):
        self.cash = torch.tensor(get_cash(), dtype=torch.float32).cuda()
        self.lives = torch.tensor(get_lives(), dtype=torch.float32).cuda()
        self.round = torch.tensor(get_round(), dtype=torch.float32).cuda()    
        # Combine all elements into a single state tensor
        self.state = torch.cat((self.grid.view(-1), self.cash, self.lives, self.round)).cuda() 

    def reset_state(self):
        # Initialize grid as a tensor
        self.grid  = torch.zeros((self.col_num, self.row_num, self.features_per_cell), dtype=torch.float32).cuda() 
        # Other state components as tensors
        self.cash = torch.tensor(get_cash(), dtype=torch.float32).cuda() 
        self.lives = torch.tensor(get_lives(), dtype=torch.float32).cuda() 
        self.round = torch.tensor(get_round(), dtype=torch.float32).cuda()      
        # Combine all elements into a single state tensor
        self.state = torch.cat((self.grid.view(-1), self.cash, self.lives, self.round)).cuda() 
        return self.state
    def get_total_action_size(self):
        place_tower_actions = self.grid.size(0) * self.grid.size(1) * len(towers) # 25x25 grid * N towers
        upgrade_tower_actions = self.grid.size(0) * self.grid.size(1) * 2 # 25x25 grid * 2 Upgrade Paths
        start_next_round_actions = 1 #Action to begin the next round
        total_action_size = place_tower_actions + upgrade_tower_actions + start_next_round_actions
        return total_action_size

    def start_next_round(self):
        '''
        Starts Next Round using the "Space Bar" hotkey
        '''
        current_round = get_round()
        #Pressed Twice to Enable Double Speed. Only Need Once
        if current_round == 0:
            pyautogui.press("space")
            pyautogui.press("space")
            reward = 1
        else:
            reward = 0
            pyautogui.press("space")
            updated_round = get_round()
            time.sleep(3)
            if updated_round == current_round:
                pyautogui.press("space")
            else:
                if updated_round < 10:
                    reward = 1# updated_round
                elif updated_round < 31:
                    reward = 1.1*1#updated_round
                elif updated_round < 32:
                    reward = 5*1#updated_round # Really struggles with round 33 due to Lead Balloons. Encourages bomb towers
                elif updated_round < 34:
                    reward = 1.2*1#updated_round
                elif updated_round < 40:
                    reward = 1.5*1#updated_round
                elif updated_round >= 40:
                    reward = 2*1 #updated_round
            if self.live_count != get_lives():
                reward = 0 # Reset Reward if Lives Lost to Encourage Saving Lives
                if (self.live_count - get_lives()) < 5.0:
                    reward -= self.live_count - get_lives()
                elif (self.live_count - get_lives()) < 15.0:
                    reward -= (self.live_count - get_lives())*2
                elif (self.live_count - get_lives()) >= 15.0:
                    reward -= (self.live_count - get_lives())*3
                self.live_count = get_lives()

        return reward
    
    
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

