'''
This script is used to read the Bloons TD5 memory to retrieve values for Cash, Lives, and Round Number due to the lack of official API.
'''

from pymem import *
from pymem.process import *
import numpy as np

# The Running Game Process
pm = pymem.Pymem("BTD5-Win.exe")

shortcut='c'
gameModule = module_from_name(pm.process_handle, "BTD5-Win.exe").lpBaseOfDll

# Static Address to use as the base for pointer locating
baseAddress = gameModule + 0x009F9BE0


def GetPtrAddr(base, offsets):
    '''
    Returns the Pointer Address given a base address and associated offsets
    '''
    addr = pm.read_int(base)
    for i in offsets:
        if i != offsets[-1]:
            addr = pm.read_int(addr+i)
    return addr + offsets[-1]

def get_cash():
    '''
    Returns a 1-D np Array of the Current Cash Value
    '''
    offsets = [0xC0, 0x90] # Offsets for Current Cash Pointer
    pointerAddr = GetPtrAddr(baseAddress, offsets)
    outputPointerAddr = np.array([int(pm.read_double(pointerAddr))])
    return outputPointerAddr
    
def get_lives():
    '''
    Returns a 1-D np Array of the Current Lives Value
    '''
    offsets = [0xC0, 0x88] # Offsets for Current Lives Pointer
    pointerAddr = GetPtrAddr(baseAddress, offsets)
    outputPointerAddr = np.array([int(pm.read_int(pointerAddr))])
    return outputPointerAddr

def get_round():
    '''
    Returns a 1-D np Array of the Current Round Value
    '''
    offsets = [0x78, 0x14] # Offsets for Current Round Pointer
    pointerAddr = GetPtrAddr(baseAddress, offsets)
    outputPointerAddr = np.array([int(pm.read_int(pointerAddr))])
    return outputPointerAddr


