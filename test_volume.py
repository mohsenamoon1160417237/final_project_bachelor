import osascript

# or
target_volume = 20
osascript.osascript("set volume output volume {}".format(target_volume))

result = osascript.osascript('get volume settings')
print(result)
print(type(result))
volInfo = result[1].split(',')
outputVol = volInfo[0].replace('output volume:', '')
print(outputVol)
