
checkVersion('SPM', 7, 0, 178)
import csv

#Coordinates of the next area
nextAreaX = 0
nextAreaY = 0

#Number of detectable cells in the area
coordinate_count = 0
file_path = 'D:/output_table.csv'

#Read coordinate points within probe detection range into script
def addposition(x,y):
    ForceSpectroscopy.addPosition(x, y)

#Read the coordinates of the center point of the next area into the script
def nextArea(x,y):
    ForceSpectroscopy.addPosition(x, y)

#Disabled platform moves
MotorizedStage.disengage()

#Clear coordinates, read new coordinates
ForceSpectroscopy.clearPositions()
#init

#Add initial position to software table, set to origin, index 0
ForceSpectroscopy.addPosition(0, 0)

#Read the coordinates of the detectable point and the center point of the next area into the script
with open(file_path, mode='r') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        x = float(row[0]) 
        y = float(row[1])
        if abs(x) < 5e-5 and abs(y) < 5e-5 :
            addposition(x, y)
            coordinate_count += 1
        else :
            nextAreaX = x
            nextAreaY = y

#Probe tip moves back to initial position
ForceSpectroscopy.moveToForcePositionIndex(0)

#Get force curves for all detectable cells and save them automatically
i = 0
for j in xrange(coordinate_count):
    i+=1
    ForceSpectroscopy.moveToForcePositionIndex(i) 
  
    Scanner.approach()
    ForceSpectroscopy.startScanning(5)
    Scanner.retractPiezo()
    Scanner.moveMotorsUp(2e-5)
    time.sleep(1.0)

#After detecting each area, the probe must return to its original position
ForceSpectroscopy.moveToForcePositionIndex(0)

#Enabling the platform
MotorizedStage.engage()
#Move the platform to the next regional center
MotorizedStage.moveToRelativePosition(nextAreaX, nextAreaY)
#Disabled platform
MotorizedStage.disengage()

#Probe down and up
Scanner.approach()
Scanner.retract()

