
from pymem import *
from pymem.process import *
import time
pm = pymem.Pymem("BTD5-Win.exe")

shortcut='c'
gameModule = module_from_name(pm.process_handle, "BTD5-Win.exe").lpBaseOfDll
offsets = [0xC0, 0x90]
offsets2 = [0xC0, 0x88]
offsets3 = [0x78, 0x14]
def GetPtrAddr(base, offsets):
    addr = pm.read_int(base)
    for i in offsets:
        if i != offsets[-1]:
            addr = pm.read_int(addr+i)
    return addr + offsets[-1]
i=0
while True:
    i = i+1
    #pm.write_int(GetPtrAddr(gameModule + 0x009F9BE0,offsets), i)
    print(pm.read_double(GetPtrAddr(gameModule + 0x009F9BE0,offsets)))
    print(pm.read_int(GetPtrAddr(gameModule + 0x009F9BE0,offsets2)))
    print(pm.read_int(GetPtrAddr(gameModule + 0x009F9BE0,offsets3))) 

    #print(i)
    time.sleep(0.1)