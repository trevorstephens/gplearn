import UNIT.GP as gp
import random
prog = gp.GP_SymReg(500,100,0.01,"all")
prog.load("puzzlebrain1program.bin")

class PuzzleOneBrain(object):
	def __init__(self):
		tabX,tabY = [],[]

	def Generate(self,randomSeed=0,longueur=6):
		random.seed(randomSeed)
		tabX,tabY = [],[None]
		tabX.append(random.randint(1,100))
		for i in range(0,longueur-1):
			tabY.append(tabX[i]+random.randint(1,100))
			tabX.append(self.calculator(tabX[-1],tabY[-1]))

		tabX[-1] = tabX[-1] - 1

		print "X: ",tabX
		print "Y: ",tabY


	def calculator(self, x, y):
		return int(prog.predict([x,y]))

POB = PuzzleOneBrain()
POB.Generate(0,7)