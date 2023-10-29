import json
import time

class Tower:
    def __init__(self, name, number, hotkey, baseCost, upgradeTableEasy, upgradeTableMed, upgradeTableHard, upgradeTableImp):
        self.name = name
        self.number = number
        self.hotkey = hotkey
        self.upgrades = (0,0)
        self.gridPosition = (0,0)
        self.difficulty = "Easy"
        self.baseCost = baseCost
        self.upgradeTableEasy = upgradeTableEasy
        self.upgradeTableMed = upgradeTableMed
        self.upgradeTableHard = upgradeTableHard
        self.upgradeTableImp = upgradeTableImp

    def get_baseCost(self):
        '''
        Returns the Tower's Base Cost According to Difficulty Scaling
        '''      
        if self.difficulty == "Easy":
            return self.baseCost[0]
        elif self.difficulty == "Medium":
            return self.baseCost[1]
        elif self.difficulty == "Hard":
            return self.baseCost[2]
        elif self.difficulty == "Impoppable":
            return self.baseCost[3]
                    

    def can_afford_placement(self, currentCash):
        '''
        Returns True if the Tower is Affordable for Placement
        '''        
        if currentCash >= self.get_baseCost():
            return True
        
        return False
    
    def get_upgrades(self):
        '''
        Returns The Current Upgrades of the Tower
        '''
        return self.upgrades
    
    def set_upgrades(self, path):
        '''
        Increments a tower's upgrades
        '''
        if path == 1:
            self.upgrades[0] += 1
        elif path == 2:
            self.upgrades[1] += 1
        
        return self.upgrades
    
    def get_upgrade_costs(self):
        '''
        Returns The Price of the Towers Next Upgrades
        '''
        upgrP1, upgrP2 = self.get_upgrades()
        upgrTable = self.get_upgrade_table()
        if upgrP1 >= 3 and upgrP2 == 2:
            upgrP1 = upgrTable[0][upgrP1]
            upgrP2 = 10000000 # Use Big Number to signal unavailable
        elif upgrP1 == 2 and upgrP2 >= 3:
            upgrP1 = 10000000 # Use Big Number to signal unavailable
            upgrP2 = upgrTable[1][upgrP2]
        else:
            upgrP1 = upgrTable[0][upgrP1]
            upgrP2 = upgrTable[1][upgrP2]

        return (upgrP1,upgrP2)
    
    def get_upgrade_table(self):
        '''
        Returns The Price of the Towers Next Upgrades
        '''
        if self.difficulty == "Easy":
            return self.upgradeTableEasy
        elif self.difficulty == "Medium":
            return self.upgradeTableMed
        elif self.difficulty == "Hard":
            return self.upgradeTableHard
        elif self.difficulty == "Impoppable":
            return self.upgradeTableImp
        return 'Yes'

    def can_afford_upgrade(self, currentCash):
        '''
        Returns a 1x2 tuple of Booleans if both tower upgrades are affordable
        '''
        upgrCostP1, upgrCostP2 = self.get_upgrade_costs()

        if currentCash >= upgrCostP1 and currentCash >= upgrCostP2:
            return (True,True)
        elif currentCash >= upgrCostP1 and currentCash < upgrCostP2:
            return (True,False)
        elif currentCash < upgrCostP1 and currentCash >= upgrCostP2:
            return (False,True)
        else:
            return (False,False)

    def get_grid_pos(self):
        '''
        Returns Grid Position of the Tower
        '''
        return self.gridPosition

    def calculate_total_upgrade_cost(self, difficulty, path):
        '''
        Returns Total Cost of a Tower's Upgrade Path
        '''
        if difficulty == 'Easy':
            table = self.upgradeTableEasy
        elif difficulty == 'Med':
            table = self.upgradeTableMed
        elif difficulty == 'Hard':
            table = self.upgradeTableHard
        elif difficulty == 'Imp':
            table = self.upgradeTableImp
        else:
            return "Invalid difficulty level"

        return sum(table[path])
    
    def get_hotkey(self):
        '''
        Returns Hotkey for the Tower
        '''
        return self.hotkey
    
def tower_from_dict(d):
    '''
    Function to Create a Tower Object from a dictionary
    '''
    return Tower(
        d["name"], 
        d["number"], 
        d["hotkey"], 
        d["baseCost"], 
        d["upgradeTableEasy"], 
        d["upgradeTableMed"], 
        d["upgradeTableHard"], 
        d["upgradeTableImp"]
    )



