
import numpy as np
import random
import math

showNothing = True

class connectionHistory():

    def __init__(self, From, To, Inno, InnoNums):
        self.fromNode = From
        self.toNode = To
        self.innovationNumber = Inno
        self.innovationNumbers = InnoNums
    
    def matches(self, genome, From, To):
        if len(genome.genes) == len(self.innovationNumbers):
            if self.From.number == self.fromNode and To.number == self.toNode:
                for i in range(len(genome.genes)):
                    if not self.innovationNumbers.contains(genome.genes[i].innovationNum):
                        return False
                return True
        return False


class connectionGene():

    def __init__(self, From, To, W, Inno):
        self.fromNode = From
        self.toNode = To
        self.weight = W
        self.innovationNum = Inno

    def mutateWeight(self):
        rand = random.randrange(0, 1)
        if rand < 0.1:
            self.weight = random.randrange(-1, 1)
        else:
            self.weight += random.gauss(1,1) / 50
        if self.weight > 1:
            self.weight = 1
        if self.weight < -1:
            self.weight = -1


class Node():
    
    def __init__(self, num):
        self.number = num
        self.input_sum = 0
        self.outputValue = 0
        self.outputConnections = list()
        self.layer = 0

    def engage(self):
        if self.layer != 0:
            self.outputValue = 1 / (1 + np.exp(-self.input_sum)) # sigmoid
        for i in range(len(self.outputConnections)):
            if self.outputConnections[i].enabled:
                self.outputConnections[i].toNode.input_sum += self.outputConnections[i].weight * self.outputValue
    
    def isConnectedTo(self, node):
        if node.layer == self.layer:
            return False
        if node.layer < self.layer:
            for i in range(len(node.outputConnections)):
                if node.outputConnections[i].toNode == self:
                    return True
        else:
            for i in range(len(self.outputConnections)):
                if self.outputConnections[i].toNode == node:
                    return True
        return False
        

class Species():
    
    def __init__(self, p):
        self.players = list()
        self.bestFitness = 0
        self.champ = None
        self.averageFitness = 0
        self.staleness = 0
        self.rep = None

        self.excessCoeff = 1
        self.weightDiffCoeff = 0.5
        self.compatibilityThreshold = 3

        self.players.append(p)
        self.bestFitness = p.fitness
        self.rep = p.brain
        self.champ = p.cloneForReplay()

    def sameSpecies(self, g):
        compatibility = 0.0
        excessAndDisjoint = self.getExcessDisjoint(g, self.rep)
        averageWeightDiff = self.averageWeightDiff(g, self.rep)

        largeGenomeNormaliser = len(g.gene) - 20
        if largeGenomeNormaliser < 1:
            largeGenomeNormaliser = 1
        
        compatibility = (self.excessCoeff * excessAndDisjoint/largeGenomeNormaliser) + (self.weightDiffCoeff * averageWeightDiff)
        return self.compatibilityThreshold > compatibility

    def addToSpecies(self, p):
        self.players.append(p)

    def getExcessDisjoint(self, brain1, brain2):
        matching = 0.0
        for i in len(brain1.genes):
            for j in len(brain2.genes):
                if brain1.genes[i].innovationNum == brain2.genes[j].innovationNum:
                    matching += 1
                    totalDiff += abs(brain1.genes[i].weight - brain2.genes[j].weight)
                    break
        return len(brain1.genes) + len(brain2.genes) - 2 * matching

    def averageWeightDiff(self, brain1, brain2):
        if len(brain1.genes) == 0 or len(brain2.genes) == 0:
            return 0
        matching = 0
        totalDiff = 0
        for i in range(len(brain1.genes)): 
            for j in range(len(brain2.genes)): 
                if brain1.genes[i].innovationNum == brain2.genes[j].innovationNum:
                    matching += 1
                    totalDiff += abs(brain1.genes[i].weight - brain2.genes[j].weight)
                    break
        return 100 if matching == 0 else totalDiff/matching
        
    def sortSpecies(self):
        temp = list()
        for i in range(len(self.players)):
            max_ = 0
            maxIndex = 0
            for j in range(len(self.players)):
                if self.players[i].fitness > max_:
                    max_ = self.players[j].fitnes
                    maxIndex = j
            temp.append(self.players[maxIndex])
            self.players.remove(maxIndex)
            i -= 1
        
        self.players = temp
        if len(self.players) == 0:
            self.staleness = 200
            return
        
        if self.players[0].fitness > self.bestFitness:
            self.staleness = 0
            self.bestFitness = self.players[0].fitness
            self.rep = self.players[0].brain
            self.champ = self.players[0].cloneForReplay
        else:
            self.staleness += 1

    def setAverage(self):
        sum_ = 0
        for i in range(len(self.players)):
            sum_ += self.players[i].fitness
        self.averageFitness = sum_ / len(self.players)

    def makeChild(self, innovationHistory):
        if random.randrange(1) < 0.25:
            child = self.selectPlayer()
        else:
            parent1 = self.selectPlayer()
            parent2 = self.selectPlayer()
            if parent1.fitness < parent2.fitness:
                child = parent2.crossover(parent1)
            else:
                child = parent1.crossover(parent2)
        child.brain.mutate(innovationHistory)
        return child

    def selectPlayer(self):
        fitnessSum = 0
        for i in range(len(self.players)):
            fitnessSum += self.players[i].fitness
        
        rand = random.randrange(fitnessSum)
        runningSum = 0
        for i in range(len(self.players)):
            runningSum += self.players[i].fitness
            if runningSum > rand:
                return self.players[i]
        return self.players[0]

    def cull(self):
        if len(self.players) > 2:
            for i in range(len(self.players)/2):
                self.players.remove(i)
                i -= 1
    
    def fitnessSharing(self):
        for i in range(len(self.players)):
            self.players[i].fitness /= len(self.players)


class Population():

    def __init__(self, size):
        self.population = list()
        self.bestPlayer = None
        self.bestScore = 0
        self.gen = 0
        self.innovationHistory = list()
        self.genPlayers = list()
        self.species = list()

        self.massExtinctionEvent = False
        self.newStage = False
        self.populationLife = 0

        for i in range(size):
            self.population.append(Player())
            self.population[i].brain.generateNetwork()
            self.population[i].brain.mutate(self.innovationHistory)

    def updateAlive(self):
        self.populationLife += 1
        for i in range(len(self.population)):
            if not self.population[i].dead:
                self.population[i].look()
                self.population[i].think()
                self.population[i].update()
                if not showNothing:
                    self.population[i].show()
    
    def done(self):
        for i in range(len(self.population)):
            if not self.population[i].dead:
                return False
        return True

    def setBestPlayer(self):
        tempBest = self.species[0].players[0]
        tempBest.gen = self.gen
        if tempBest.score > self.bestScore:
            self.genPlayers.append(tempBest.cloneForReplay())
            print('old best:', self.bestScore)
            print('new best:', tempBest.score)
            self.bestScore = tempBest.score
            self.bestPlayer = tempBest.cloneForReplay()

    def naturalSelection(self):
        self.speciate()
        self.calculateFitness()
        self.sortSpecies()
        if self.massExtinctionEvent:
            self.massExtinction()
            self.massExtinctionEvent = False
        self.cullSpecies()
        self.setBestPlayer()
        self.killStaleSpecies()

        averageSum = self.getAvgFitnessSum()
        children = list()
        print('species:')
        for i in range(len(self.species)):
            children.append(self.species[i].champ)
            childrenNum = math.floor(self.species[j].averageFitness/averageSum * len(self.population)) - 1
            for _ in range(len(childrenNum)):
                children.append(self.species[i].makeChild(self.innovationHistory))
        while len(children) < len(self.population):
            children.append(self.species[0].makeChild(self.innovationHistory))
        self.population.clear()
        self.population = children
        self.gen += 1
        for i in range(len(self.population)):
            self.population[i].brain.generateNetwork()
        self.populationLife = 0

    def speciate(self):
        for s in self.species:
            s.players.clear()
        for i in range(len(self.population)):
            speciesFound = False
            for s in self.species:
                if s.sameSpecies(self.population[i].brain):
                    s.addToSpecies(self.population[i])
                    speciesFound = True
                    break
            if not speciesFound:
                self.species.append(Species(self.population[i]))
    
    def calculateFitness(self):
        for p in self.population:
            p.calculateFitness()

    def sortSpecies(self):
        for s in self.species:
            s.sortSpecies()
        temp = list()
        for i in range(len(self.species)):
            max_ = 0
            maxIndex = 0
            for index, s in enumerate(self.species):
                if s.bestFitness > max_:
                    max_ = s.bestFitness
                    maxIndex = index
            temp.append(self.species[maxIndex])
            self.species.remove(maxIndex)
            i -= 1
        self.species = temp

    def killStaleSpecies(self):
        for i in range(2, len(self.species)):
            if self.species[i].staleness >= 15:
                self.species.remove(i)
                i -= 1
    
    def getAvgFitnessSum(self):
        averageSum = 0
        for s in self.species:
            averageSum += s.averageFitness
        return averageSum

    def cullSpecies(self):
        for s in self.species:
            s.cull()
            s.fitnessSharing()
            s.setAverage()

    def massExtinction(self):
        for i in range(5, len(self.species)):
            self.species.remove(i)
            i -= 1


class Genome():

    def __init__(self, In, Out):
        self.genes = list()
        self.nodes = list()
        self.inputs = In
        self.outputs = Out
        self.layers = 2
        self.nextNode = 0
        self.biasNode = 0
        self.network = list()
        for i in range(self.inputs):
            self.nodes.append(Node(i))
            self.nextNode += 1
            self.nodes[i].layer = 0
        for i in range(self.outputs):
            self.nodes.append(Node(i+self.inputs))
            self.nextNode += 1
            self.nodes[i+self.inputs].layer = 1
        self.nodes.append(Node(self.nextNode))
        self.biasNode = self.nextNode
        self.nextNode += 1
        self.nodes[self.biasNode].layer = 0

    def getNode(self, nodeNumber):
        for n in self.nodes:
            if n.number == nodeNumber:
                return n
        return None
    
    def connectNodes(self):
        for n in self.nodes:
            n.outputConnections.clear()
        for g in self.genes:
            g.fromNode.outputConnections.append(g)

    def feedForward(self, inputValues):
        for i in range(self.inputs):
            self.nodes[i].outputValue = inputValues[i]
        self.nodes[self.biasNode].outputValue = 1
        for n in self.network:
            n.engage()
        outs = list()
        for i in range(self.outputs):
            outs.append(self.nodes[i+self.inputs].outputValue)
        for i in range(len(self.nodes)):
            self.nodes[i].input_sum = 0
        return outs

    def generateNetwork(self):
        self.connectNodes()
        self.network = list()
        for l in range(self.layers):
            for i in self.nodes:
                if i.layer == l:
                    self.network.append(i)

    def addNode(self, innovationHistory):
        if len(self.genes) == 0:
            self.addConnection(innovationHistory)
            return
        randomConnection = math.floor(random.random(len(self.genes)))
        while self.genes[randomConnection].fromNode == self.nodes[self.biasNode] and len(self.genes) != 1:
            randomConnection = math.floor(random.random(len(self.genes)))
        self.genes[randomConnection].enabled = False

        newNodeNum = self.nextNode
        self.nodes.append(Node(newNodeNum))
        self.nextNode += 1
        connectionInnovationNumber = self.getInnovationNumber(innovationHistory, self.genes[randomConnection].fromNode, self.getNode(newNodeNum))
        self.genes(connectionGene(self.nodes[self.biasNode], self.getNode(newNodeNum), 0, connectionInnovationNumber))

        if self.getNode(newNodeNum).layer == self.genes[randomConnection].toNode.layer:
            for i in range(len(self.nodes) - 1):
                if self.nodes[i].layer >= self.getNode(newNodeNum).layer:
                    self.nodes[i].layer += 1
            self.layers += 1
        self.connectNodes()

    def addConnection(self, innovationHistory):
        if self.fullyConnected():
            print('Connection failed')
            return
        randomNode1 = random.randint(len(self.nodes))
        randomNode2 = random.randint(len(self.nodes))
        while self.checkNodes(randomNode1, randomNode2)):
            randomNode1 = random.randint(len(self.nodes))
            randomNode2 = random.randint(len(self.nodes))
        if self.nodes[randomNode1].layer > self.nodes[randomNode2].layer:
            temp = randomNode2
            randomNode2 = randomNode1
            randomNode1 = temp
        connectionInnovationNumber = self.getInnovationNumber(innovationHistory, self.nodes[randomNode1], self.nodes[randomNode2])
        self.genes.append(connectionGene(self.nodes[randomNode1], self.nodes[randomNode2], random.random(-1, 1), connectionInnovationNumber))
        self.connectNodes()
        
    def checkNodes(self, r1, r2):
        if self.nodes[r1].layer == self.nodes[r2].layer):
            return True
        if self.nodes[r1].isConnectedTo(self.nodes[r2]):
            return True
        return False

    def getInnovationNumber(self, innovationHistory, From, To):
        isNew = True
        connectionInnovationNumber = nextConnectionNum
        for i in innovationHistory:
            if i.matches(self, From, To):
                isNew = False
                connectionInnovationNumber = i.innovationNumber
                break
        if isNew:
            innoNumbers = list()
            for i in self.genes:
                innoNumbers.append(i.innovationNum)
            self.innovationHistory.append(connectionHistory(From.number, To.number, connectionInnovationNumber, innoNumbers))
            self.nextConnectionNum += 1
        return connectionInnovationNumber

    def fullyConnected(self):
        maxConnections = 0
        nodesInLayer = list()
        for i in self.nodes:
            nodesInLayer[i.layer] += 1
        for i in range(len(self.layers) - 1):
            nodesInFront = 0
            for j in range(i+1, self.layers):
                nodesInFront += nodesInLayer[j]
            maxConnections += nodesInLayer[i] * nodesInFront
        if maxConnections == len(self.genes):
            return True
        return False

    def mutate(self, innovationHistory):
        if len(self.genes) == 0:
            self.addConnection(innovationHistory)
        rand1 = random.random(0, 1)
        if rand1 < 0.8:
            for i in self.genes:
                i.mutateWeight()
        rand2 = random.random(0, 1)
        if rand2 < 0.08:
            self.addConnection(innovationHistory)
        rand3 = random.random(0, 1)
        if rand3 < 0.02:
            self.addNode(innovationHistory)

    def crossover(self, parent2):
        child = Genome(self.inputs, self.outputs, True)
        child.genes.clear()
        child.nodes.clear()
        child.layers = self.layers
        child.nextNode = self.nextNode
        child.biasNode = self.biasNode
        childGenes = list()
        isEnabled = list()

        for i in range(len(self.genes)):
            setEnabled = True
            parent2gene = self.matchingGene(parent2, self.genes[i].innovationNumber)
            if parent2gene != -1:
                if not self.genes[i].enabled or not parent2.genes[parent2gene].enabled:
                    if random.random(0, 1) < 0.75:
                        setEnabled = False
                rand = random.random(0, 1)
                if rand < 0.5:
                    childGenes.append(self.genes[i])
                else:
                    childGenes.add(parent2.genes[parent2gene])
            else:
                childGenes.append(self.genes[i])
                setEnabled = self.genes[i].enabled
            isEnabled.append(setEnabled)
        for i in self.nodes:
            child.nodes.append(i)
        for i in range(len(childGenes)):
            child.genes.append(childGenes[i].clone(child.getNode(childGenes[i].fromNode.number), child.getNode(childGenes[i].toNode.number)))
            child.genes[i].enabled = isEnabled[i]
        child.connectNodes()
        return child

    def matchingGene(self, parent2, innovationNumber):
        for i in range(len(parent2.genes)):
            if parent2.genes[i].innovationNum == innovationNumber:
                return i
        return -1


