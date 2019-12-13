import random
import sys
import math
import pygame
from pygame.locals import *

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

START_MUTATION_RATE = 0.1
MIN_MUTATION_RATE = 0.05
MUTATION_STEP = 0.05
POPULATION_SIZE = 50
FPS = 30
SCREENWIDTH  = 288
SCREENHEIGHT = 512
PIPEGAPSIZE  = 100 # gap between upper and lower part of pipe
BASEY        = SCREENHEIGHT * 0.79
# image and hitmask  dicts
IMAGES, HITMASKS = {}, {}

# list of all possible players (tuple of 3 positions of flap)
PLAYERS_LIST = (
    # red bird
    (
        'assets/sprites/redbird-midflap.png',
    ),
    # blue bird
    (
        'assets/sprites/bluebird-midflap.png',
    ),
    # yellow bird
    (
        'assets/sprites/yellowbird-midflap.png',
    ),
)
# list of backgrounds
BACKGROUNDS_LIST = (
    'assets/sprites/background-day.png',
    'assets/sprites/background-night.png',
)
# list of pipes
PIPES_LIST = (
    'assets/sprites/pipe-green.png',
    'assets/sprites/pipe-red.png',
)


# ---------------------------------------------------------------------------------------------------------------------
class Brain:
    def __init__(self,genomeInputs=7,genomeOutputs=1):
        model = Sequential()
        model.add(Dense(genomeOutputs, activation='sigmoid', input_dim=genomeInputs))
        # model.add(Dense(genomeOutputs, activation='sigmoid')) # Output (to flap or no)
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss="mse", optimizer=sgd, metrics=["accuracy"])
        self.model = model
        Brain.randomize(self)
        self.numInputs = genomeInputs
        self.numOutputs = genomeOutputs

    def randomize(self):
        weights = self.model.get_weights()
        config = self.model.get_config()
        for xi in range(len(weights)):
            for yi in range(len(weights[xi])):
                change = random.uniform(-1,1)
                weights[xi][yi] = change
        self.model.set_weights(weights)

    def predict(self,inputs):
        npIN = np.asarray(inputs)
        npIN2 = np.atleast_2d(npIN)
        outputProb = self.model.predict(npIN2,1)
        if outputProb <= 0.5:
            return True
        return False

    def mutate(self,mutationRate):
        # 10% of the time that we are mutating, completely replace weight. Else slightly change it.  Keep between (-1,1)
        weights = self.model.get_weights()
        for xi in range(len(weights)):
            for yi in range(len(weights[xi])):
                if random.uniform(0, 1) < mutationRate:
                    if random.uniform(0,1) < 0.1:
                        # Replace weight
                        weights[xi][yi] = random.uniform(-1,1)
                    else: # Shift weight a bit
                        change = random.gauss(0,1) / 50
                        weights[xi][yi] += change
                        if weights[xi][yi] > 1:
                            weights[xi][yi] = 1
                        elif weights[xi][yi] < -1:
                            weights[xi][yi] = -1
        self.model.set_weights(weights)

    def clone(self):
        clone = Brain(self.numInputs,self.numOutputs)
        clone.model.set_weights(self.model.get_weights())
        return clone

    @staticmethod
    def crossover(brain1,brain2):
        xover = Brain(brain1.numInputs,brain1.numOutputs)
        xoverweight = xover.model.get_weights()
        weight1 = brain1.model.get_weights()
        weight2 = brain2.model.get_weights()

        for xi in range(len(xoverweight)):
            for yi in range(len(xoverweight[xi])):
                if random.uniform(0, 1) < 0.5:
                    xoverweight[xi][yi] = weight1[xi][yi]
                else:
                    xoverweight[xi][yi] = weight2[xi][yi]
        xover.model.set_weights(xoverweight)
        return xover


# ---------------------------------------------------------------------------------------------------------------------

class Bird:
    def __init__(self, x=SCREENWIDTH*0.2,y=SCREENHEIGHT/2):
        # Basic Player things
        self.x = x
        self.y = y
        self.velY=0
        self.velX = -4
        self.isOnGround = False
        self.dead = False
        self.score = 0

        # player velocity, max velocity, downward accleration, accleration on flap
        # Current real-time values
        self.playerVelY    =  -9   # player's velocity along Y, default same as playerFlapped
        self.playerAccY    =   1   # players downward accleration
        self.playerRot     =  45   # player's rotation
        self.playerFlapped = False # True when player flaps

        # limits, max values, etc. Physics
        self.playerMaxVelY =  10   # max vel along Y, max descend speed
        self.playerMinVelY =  -8   # min vel along Y, max ascend speed
        self.playerVelRot  =   3   # angular speed
        self.playerRotThr  =  20   # rotation threshold
        self.playerFlapAcc =  -9   # players speed on flapping
        self.visibleRot = 0        # make bird look less stupid

        # AI stuffs/ Genetic Algorithm
        self.fitness = 0
        self.vision = [] #Input to neural Net
        self.decision = [] # Output of NN
        self.unadjustedFitness = 0
        self.lifespan = 0 # How long player lived for self.fitness
        self.bestScore = 0 # Store self.score achieved for replay
        self.score = 0
        self.gen = 0
        self.isBest = False
        self.genomeInputs = 7
        self.genomeOutputs = 1
        self.brain = Brain(self.genomeInputs,self.genomeOutputs)

    def predict(self, upperPipes, lowerPipes):
        self.defineInputVec(upperPipes,lowerPipes)
        yesFlap = self.brain.predict(self.vision)
        if yesFlap:
            self.flap()

    def flap(self):
        if self.y > -2 * IMAGES['player'][0].get_height():
            self.playerVelY = self.playerFlapAcc
            self.playerFlapped = True

    def defineInputVec(self,upperPipes,lowerPipes):
        hDist = self.horizontalDistToNextPipe(upperPipes)
        uDist = self.distToUpperPipe(upperPipes)
        lDist = self.distToLowerPipe(lowerPipes)
        gDist = self.distToGround()
        vVelo = self.playerVelY
        self.vision = [hDist[0],hDist[2],uDist,lDist,gDist,vVelo,1.0]

    def updateScore(self,upperPipes):
        playerMidPos = self.x + IMAGES['player'][0].get_width() / 2
        for pipe in upperPipes:
            pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                self.score += 1

    def horizontalDistToNextPipe(self,upperPipes):
        minDist = 99999
        distToFront = 999999
        minIdx = 0
        pipeWidth = IMAGES['pipe'][0].get_width()
        playerWidth = IMAGES['player'][0].get_width()
        for i in range(len(upperPipes)):
            pipe = upperPipes[i]
            dist = (pipe['x']+pipeWidth) - self.x
            dTF = pipe['x'] - (self.x + playerWidth)
            if (dist > 0) and (dist < minDist):
                minDist = dist
                distToFront = dTF
                minIdx = i
        return [minDist,minIdx,distToFront]

    def distToGround(self):
        playerHeight = IMAGES['player'][0].get_height()
        dist = (BASEY - 1) - (self.y + playerHeight)
        return dist

    def distToUpperPipe(self,upperPipes):
        dist = self.horizontalDistToNextPipe(upperPipes)
        idx = dist[1]
        pipe = upperPipes[idx]
        pipeH = IMAGES['pipe'][0].get_height()
        distY = self.y - (pipe['y'] + pipeH)
        return distY

    def distToLowerPipe(self,lowerPipes):
        dist = self.horizontalDistToNextPipe(lowerPipes)
        idx = dist[1]
        pipe = lowerPipes[idx]
        playerH = IMAGES['player'][0].get_height()
        distY = pipe['y'] - (self.y + playerH)
        return distY

    def rotate(self):
        if self.playerRot > -90:
            self.playerRot -= self.playerVelRot

        self.visibleRot = self.playerRotThr
        if self.playerRot <= self.playerRotThr:
            self.visibleRot = self.playerRot

    def update(self,upperPipes,lowerPipes):
        if not self.dead:
            self.lifespan += 1
            self.checkCrash(upperPipes,lowerPipes)

            if not self.dead:
                # Decide to flap
                self.predict(upperPipes,lowerPipes) # calls self.flap() if flapping

                # check for score
                self.updateScore(upperPipes)

                # rotate the player
                self.rotate()

                # player's movement
                self.move()
                return False
            else:
                return True

    def show(self):
        if not self.dead:
            if self.isBest:
                playerSurface = pygame.transform.rotate(IMAGES['player'][1], self.visibleRot)
                SCREEN.blit(playerSurface, (self.x, self.y))
            else:
                playerSurface = pygame.transform.rotate(IMAGES['player'][0], self.visibleRot)
                SCREEN.blit(playerSurface, (self.x, self.y))
            return True
        return False

    def move(self):
        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False

            # more rotation to cover the threshold (calculated in visible rotation)
            self.playerRot = 45
        playerHeight = IMAGES['player'][0].get_height()
        self.y += min(self.playerVelY, BASEY - self.y - playerHeight)

    def calculateFitness(self):
        self.fitness = 1 + self.score * self.score + self.lifespan / 20.0

    def checkCrash(self,upperPipes,lowerPipes):
        """returns True if player collders with base or pipes."""
        playerWidth = IMAGES['player'][0].get_width()
        playerHeight = IMAGES['player'][0].get_height()

        # if player crashes into ground
        if self.y + playerHeight >= BASEY - 1:
            self.dead = True
            self.isOnGround = True
            return True
        else:
            playerRect = pygame.Rect(self.x, self.y,
                          playerWidth, playerHeight)
            pipeW = IMAGES['pipe'][0].get_width()
            pipeH = IMAGES['pipe'][0].get_height()

            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                # upper and lower pipe rects
                uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
                lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

                # player and upper/lower pipe hitmasks
                pHitMask = HITMASKS['player'][0]
                uHitmask = HITMASKS['pipe'][0]
                lHitmask = HITMASKS['pipe'][1]

                # if bird collided with upipe or lpipe
                uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
                lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

                if uCollide or lCollide:
                    self.dead = True
                    return True
        return False

    def gimmieBaby(self,parent2):
        baby = Bird()
        baby.brain = self.brain.crossover(self.brain.clone(),parent2.brain)
        return baby

# ---------------------------------------------------------------------------------------------------------------------

class Population:
    def __init__(self, size=100, dotStartX=0, dotStartY=0):
        self.birds = []
        for i in range (size):
            self.birds.append(Bird(dotStartX,dotStartY))
        self.fitnessSum = 0
        self.generation = 1
        self.bestBird = 0
        # self.bestSteps = len(self.dots[0].brain.directions)
        self.avgFitness = 0
        self.stdDevFitness = 0
        self.maxFitness = 0
        self.currentBestScore = 0

    def show(self):
        maxShow = 20
        for i in range (1,len(self.birds)):
            if maxShow > 0:
                if self.birds[i].show():
                    maxShow -= 1
        self.birds[0].show()

    def update(self,upperPipes,lowerPipes):
        for i in range (len(self.birds)):
            self.birds[i].update(upperPipes,lowerPipes)

    def findMaxScore(self):
        self.currentBestScore = 0
        for bird in self.birds:
            if self.currentBestScore < bird.score:
                self.currentBestScore = bird.score

    def calculateAvgFitness(self):
        self.avgFitness = self.fitnessSum / len(self.birds)

    def calculateStdDevFitness(self):
        runningSum = 0
        for bird in self.birds:
            runningSum += math.pow(bird.fitness - self.avgFitness,2)
        self.stdDevFitness = math.sqrt(runningSum/len(self.birds))

    def printStats(self):
        #print("Generation: ",self.generation)
        print("Best Fit:   ",self.maxFitness)
        print("Best Score: ",self.currentBestScore)
        print("Mean Fit:   ",self.avgFitness)
        print("StdDev Fit: ",self.stdDevFitness)
        print(" ")

    def naturalSelection(self):
        # Print stats of the generation
        Population.setBestBird(self)
        Population.calculateFitnessSum(self)
        Population.calculateAvgFitness(self)
        Population.calculateStdDevFitness(self)
        Population.printStats(self)

        # Generate new dot list (next generation)
        newBirds = []
        newBirds.append(self.birds[self.bestBird].gimmieBaby(self.birds[self.bestBird]))
        newBirds[0].isBest = True
        for i in range(1,len(self.birds)):
            # Select Parent based on fitness
            parent1 = Population.selectParent(self)
            parent2 = Population.selectParent(self)

            # Get baby from them
            baby = parent1.gimmieBaby(parent2)
            newBirds.append(baby)
        self.birds = newBirds.copy()
        self.generation += 1

    def mutateDemBabies(self,mutationRate):
        for i in range(1,len(self.birds)):
            self.birds[i].brain.mutate(mutationRate)

    def setBestBird(self):
        maxScore = 0
        maxIdx = 0
        for i in range (len(self.birds)):
            if self.birds[i].fitness > maxScore:
                maxScore = self.birds[i].fitness
                maxIdx = i
        self.bestBird = maxIdx
        self.maxFitness = maxScore

        # if self.dots[self.bestDot].reachedGoal:
        #     self.bestSteps = self.dots[self.bestDot].brain.step

    def calculateFitness(self):
        for i in range (len(self.birds)):
            self.birds[i].calculateFitness()

    def calculateFitnessSum(self):
        self.fitnessSum = 0
        for bird in self.birds:
            self.fitnessSum += bird.fitness

    def selectParent(self):
        rand = random.uniform(0,self.fitnessSum)
        runningSum = 0
        for bird in self.birds:
            runningSum += bird.fitness
            if runningSum > rand:
                return bird
        # Should never get to this point
        print ("HALP YOU BROKE IT - natural selection & select parent")
        return None

    def allBirdsDead(self):
        for i in range (len(self.birds)):
            if (not self.birds[i].dead):
                return False
        return True

# ---------------------------------------------------------------------------------------------------------------------
try:
    xrange
except NameError:
    xrange = range


def main():
    global SCREEN, FPSCLOCK
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
    pygame.display.set_caption('Flappy Bird')

    # numbers sprites for score display
    IMAGES['numbers'] = (
        pygame.image.load('assets/sprites/0.png').convert_alpha(),
        pygame.image.load('assets/sprites/1.png').convert_alpha(),
        pygame.image.load('assets/sprites/2.png').convert_alpha(),
        pygame.image.load('assets/sprites/3.png').convert_alpha(),
        pygame.image.load('assets/sprites/4.png').convert_alpha(),
        pygame.image.load('assets/sprites/5.png').convert_alpha(),
        pygame.image.load('assets/sprites/6.png').convert_alpha(),
        pygame.image.load('assets/sprites/7.png').convert_alpha(),
        pygame.image.load('assets/sprites/8.png').convert_alpha(),
        pygame.image.load('assets/sprites/9.png').convert_alpha()
    )

    # game over sprite
    IMAGES['gameover'] = pygame.image.load('assets/sprites/gameover.png').convert_alpha()
    # message sprite for welcome screen
    IMAGES['message'] = pygame.image.load('assets/sprites/message.png').convert_alpha()
    # base (ground) sprite
    IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

    # select random player sprites
    # randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
    IMAGES['player'] = (
        pygame.image.load(PLAYERS_LIST[2][0]).convert_alpha(),
        pygame.image.load(PLAYERS_LIST[1][0]).convert_alpha(),
    )
    # Define New Player
    startX = int(SCREENWIDTH * 0.2)
    startY = int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2)
    testPop = Population(POPULATION_SIZE,startX,startY)
    while True:
        # select random background sprites
        randBg = random.randint(0, len(BACKGROUNDS_LIST) - 1)
        IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[randBg]).convert()

        # select random pipe sprites
        pipeindex = random.randint(0, len(PIPES_LIST) - 1)
        IMAGES['pipe'] = (
            pygame.transform.flip(
                pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), False, True),
            pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
        )

        # hismask for pipes
        HITMASKS['pipe'] = (
            getHitmask(IMAGES['pipe'][0]),
            getHitmask(IMAGES['pipe'][1]),
        )

        # hitmask for player
        HITMASKS['player'] = (
            getHitmask(IMAGES['player'][0]),
        )

        #movementInfo = showWelcomeAnimation()
        mainGame(testPop)
        showGameOverScreen(testPop)


def showWelcomeAnimation():
    """Shows welcome screen animation of flappy bird"""
    # Define New Player
    playerx = int(SCREENWIDTH * 0.2)
    playery = int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2)


    player = Bird(playerx,playery)

    # draw sprites
    SCREEN.blit(IMAGES['background'], (0,0))
    SCREEN.blit(IMAGES['player'][0],
                (player.x, player.y))
    SCREEN.blit(IMAGES['base'], (0, BASEY))

    pygame.display.update()
    FPSCLOCK.tick(FPS)
    return player

def mainGame(testPop):
    # Change text to display new generation number to screen
    font = pygame.font.Font(None, 36)
    genDisp = font.render("Gen: "+str(testPop.generation), 1, (255,255,255))
    text_width, text_height = font.size("Gen: "+str(testPop.generation))
    # textpos = genDisp.get_rect()
    # textpos.centerx = IMAGES['background'].get_rect().centerx

    # get 2 new pipes to add to upperPipes lowerPipes list
    newPipe1 = getRandomPipe()
    newPipe2 = getRandomPipe()

    # list of upper pipes
    upperPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
    ]

    # list of lowerpipe
    lowerPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
    ]

    pipeVelX = -4

    while True:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            # if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
            #     player.flap()

        if testPop.allBirdsDead():
            return True

        # Update each bird (update fn checks if alive)
        for bird in testPop.birds:
            bird.update(upperPipes,lowerPipes)


        # check for crash here
        # if player.update(upperPipes,lowerPipes):
        #     return {
        #         'player': player,
        #         'upperPipes': upperPipes,
        #         'lowerPipes': lowerPipes,
        #     }


        # move pipes to left
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipe['x'] += pipeVelX
            lPipe['x'] += pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            upperPipes.append(newPipe[0])
            lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            upperPipes.pop(0)
            lowerPipes.pop(0)

        # Re-draw Everything
        SCREEN.blit(IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (0, BASEY))
        SCREEN.blit(genDisp, (SCREENWIDTH/2-text_width/2,int(SCREENHEIGHT*0.89)))
        # print score so player overlaps the score
        testPop.findMaxScore()
        showScore(testPop.currentBestScore)

        testPop.show()
        pygame.display.update()
        FPSCLOCK.tick(FPS)


def showGameOverScreen(testPop):
    """crashes the player down ans shows gameover image"""

    # Genetic Algorithm
    testPop.calculateFitness()
    testPop.naturalSelection()      #also prints out stats
    mutationRate = START_MUTATION_RATE - testPop.generation*MUTATION_STEP
    if mutationRate < MIN_MUTATION_RATE:
        mutationRate = MIN_MUTATION_RATE
    testPop.mutateDemBabies(mutationRate)
    print("Generation: ", testPop.generation)



    # player = crashInfo['player']
    # upperPipes, lowerPipes = crashInfo['upperPipes'], crashInfo['lowerPipes']
    #
    # while True:
    #     for event in pygame.event.get():
    #         if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
    #             pygame.quit()
    #             sys.exit()
    #         if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
    #             return
    #
    #     # draw sprites
    #     SCREEN.blit(IMAGES['background'], (0,0))
    #
    #     for uPipe, lPipe in zip(upperPipes, lowerPipes):
    #         SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
    #         SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))
    #
    #     SCREEN.blit(IMAGES['base'], (0, BASEY))
    #     showScore(player.score)
    #
    #     playerSurface = pygame.transform.rotate(IMAGES['player'][0], 0)
    #     SCREEN.blit(playerSurface, (player.x,player.y))
    #     SCREEN.blit(IMAGES['gameover'], (50, 180))
    #
    #     FPSCLOCK.tick(FPS)
    #     pygame.display.update()


def getRandomPipe():
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapY = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
    gapY += int(BASEY * 0.2)
    pipeHeight = IMAGES['pipe'][0].get_height()
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE}, # lower pipe
    ]

def showScore(score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0 # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()

def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in xrange(rect.width):
        for y in xrange(rect.height):
            if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                return True
    return False

def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in xrange(image.get_width()):
        mask.append([])
        for y in xrange(image.get_height()):
            mask[x].append(bool(image.get_at((x,y))[3]))
    return mask

if __name__ == '__main__':
    main()
