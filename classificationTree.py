from math import log
import pandas as pd
import numpy as np
import copy

class internalNode: 
    #node carries details of the class that is split and on what basis it is split
    def __init__(self,parent=None,featureDecision='',splittingCondition=None):
        
        self.parent=parent
        self.featureDecision=featureDecision # string which is the name of the feature on the basis of which we are splitting the dataset
        self.splittingCondition=splittingCondition # list containing all the possibiltiies(2)  under featureDecision based on which it will classify 
        self.leftChild=None
        self.rightChild=None

class decisionNode:
    #carries either of the decisions that is possible

    def __init__(self,parent=None,featureDecision='') :
        self.parent=parent
        self.featureDecision=featureDecision
        self.leftChild=None
        self.rightChild=None



class classTrees:
#this will contain the tree itself and all the necessary functions to build it
    def __init__(self,file='train.csv',target=None,splitpercentage=70): 
        #splitpercentage determines how much of the dataset will be attributed for training and testing
        #target (or self.target) is a string reference to the column that the user selects as the target column (y values)
        #by default, the last index is taken
        self.df=pd.read_csv(file)

        self.depth=3
        
        if target!=None:
            self.targetName=target
        else:
            self.targetName=self.df.columns[self.df.shape[1]-1]
        
        

        self.dataShell=pd.DataFrame()
        self.root=internalNode()
        self.contSafe(self.df) #converts binary features into 1 and 0 for gini to interpret
        self.dataAlloc(self.df,splitpercentage)

        
        self.predictions=[]

    def contSafe(self,dataset):
        for col in dataset:
            if dataset.drop_duplicates(subset=col).shape[0]==2 and col!=self.targetName:
                values=list(dataset.drop_duplicates(subset=col)[col])

                dataset[col].replace(values[0],0,inplace=True)
                dataset[col].replace(values[1],1,inplace=True)



    def dataAlloc(self,df,splitpercentage): #dataset is split based on splitpercentage

        self.trainingData=self.df.iloc[:df.shape[0]*splitpercentage//100, :]
        testing=self.df.iloc[df.shape[0]*splitpercentage//100:, :]

        self.y_test=list(testing[self.targetName])
        self.testingData=testing.drop(self.targetName,axis=1)




    def buildTree(self,parent,current,currentDataset):  
        #this is recursive in nature

        

        if currentDataset.empty or currentDataset.shape[1]==1 or self.stopCriteria(currentDataset):

            current.featureDecision = currentDataset[self.targetName].mode().values[0]
            current.parent=parent 


        else:
            current=self.splitNode(parent,current,currentDataset)

            subLeftDataset=self.splitDataset(current,currentDataset,'l')
            subRightDataset=self.splitDataset(current,currentDataset,'r')

            current.rightChild=internalNode()
            current.leftChild=internalNode()
            
            self.buildTree(current,current.leftChild,subLeftDataset)
            self.buildTree(current,current.rightChild,subRightDataset) 
            


    def splitNode(self,parent,current,currentDataset): # tested
        #this will split the node ie assign it a left and right child on the basis of a splitting criteria
        
        splits=self.calcCriteria(3,currentDataset) #returns the most accurate feature for splitting
        splittingFeature=splits[0]
        print('Feature chosen:', splittingFeature)
        print()

        splittingCondition=splits[1]
        #returns a list of 2 elements of which each is a possible element of the feature
        

        current.parent=parent
        current.featureDecision=splittingFeature
        current.splittingCondition=splittingCondition


        return current


    def splitDataset(self,current,currentDataset,child,drop=True) : #not tested
    # divides the dataset into 2, one meant for the left child and the other for the right child
        #accounting for reduction in size
        if child is 'l':
            return self.leftSplitDataset(current,currentDataset,drop)
        else:
            return self.rightSplitDataset(current,currentDataset,drop)


    def leftSplitDataset(self,current,currentDataset,drop) :
        
        print('LEFT')
        print("-----------------------------------------------------------------------------------------")

        if drop==True:      
            data=currentDataset[currentDataset[current.featureDecision]<current.splittingCondition].drop([current.featureDecision], axis=1)
        else:
            data=currentDataset[currentDataset[current.featureDecision]<current.splittingCondition]

        print(data)
        print()
        return data

    def rightSplitDataset(self,current,currentDataset,drop=True) : 
        #similar to leftSplitDataset()
        
        print('RIGHT')
        print("-----------------------------------------------------------------------------------------")


        if drop==True:
            data=currentDataset[currentDataset[current.featureDecision]>=current.splittingCondition].drop([current.featureDecision], axis=1)
        else:
            data=currentDataset[currentDataset[current.featureDecision]>=current.splittingCondition]

        print(data)
        print()
        return data





    def calcCriteria(self,option,dataset):


        if option is 3:
            #print('Data passed to gini:')
           #print(dataset)
            return self.giniIndexcont(dataset)

        
        else:
            return None


    def gini_x(self,dataset,targName, groups, class_values):
        n_instances=len(groups[0])+len(groups[1])



        
        gini = 0.0


        for g in groups:        
            
            size = float(g.shape[0])
            score = 0.0

            
            if size == 0:
                size = 1


            for class_val in class_values:
                


                if size==float(0):
                    size=1

                p = g[g[targName]==class_val].shape[0] / size
                

                score += p * p


            gini += (1.0 - score) * (size / n_instances)


        return gini
     
    def giniIndexcont(self,dataset):

        glist = []



        
        class_values =list(dataset.drop_duplicates(subset=self.targetName)[self.targetName])

 

        for col in dataset.columns:
            if col!=self.targetName: 

                for row in dataset[col]:



                    l1=dataset[dataset[col] < row ]
                    l2=dataset[dataset[col] >= row ]



                    groups = [l1,l2]
                    gini = self.gini_x(dataset,self.targetName, groups, class_values)

                    glist.append([gini,row,col])


            
        min = glist[0][0]
        sf = glist[0][2] 
        val = glist[0][1]
        for j in range(len(glist)):

            if glist[j][0] < min:
                min = glist[j][0]
                sf = glist[j][2]
                val = glist[j][1]
        

        res = [ sf , val ]
        #print(res)
        return res

        

    def stopCriteria(self,currentDataset):
        if self.purity(currentDataset) >= 80:

            return True
        return False
        

        #stops building the tree
        
    def purity(self, data):

        
        pure=data[data[self.targetName]==data[self.targetName].mode().values[0]].shape[0] / data.shape[0] * 100



        return pure

        
                
    def displayDataset(self,dataset):
        
        print(dataset)

    def preorderDisp(self,current):
        
        if current != None:
            print(current.featureDecision)
            self.preorderDisp(current.leftChild)
            self.preorderDisp(current.rightChild)

    def height(self,current) :

            if (current == None) :
                return -1

            elif (self.leaf(current)) :
                return 0

            else :

                a= self.height(current.leftChild) + 1
                b= self.height(current.rightChild) + 1


                return a if a > b else b

    def leaf(self,current):

        if ((current.leftChild == None) and (current.rightChild == None)): 
            return True


        return False

    
    def predict(self,data=None): #should return an array of predictions
        
        if data==None:
            data=self.testingData


        self.traverseTree(self.root,data)

        self.predictions=list(self.dataShell[self.targetName].sort_index()) 
        return self.predictions
    
    
    def traverseTree(self,current,currentDataset): 
        #this is recursive in nature
        
        
        if self.leaf(current):

            self.assignDecision(currentDataset,current.featureDecision)
            self.dataCombine(currentDataset)
            
        else:


            subLeftDataset=self.splitDataset(current,currentDataset,'l',drop=False)
            subRightDataset=self.splitDataset(current,currentDataset,'r',drop=False)

            
            self.traverseTree(current.leftChild,subLeftDataset)
            self.traverseTree(current.rightChild,subRightDataset)

            
    def assignDecision(self,currentDataset,featureDecision):

        currentDataset.insert(len(currentDataset.columns), self.targetName, featureDecision)

        
        
    def dataCombine(self,currentDataset):
        self.dataShell=pd.concat([self.dataShell,currentDataset])

    def accuracy(self,correctValues=None):
        if correctValues==None:
            correctValues=self.y_test

        correct=0
        for i in range(len(self.predictions)):
            if self.predictions[i]==correctValues[i]:
                correct+=1

        print("Correlation: ",correct/len(self.predictions)*100,'%')

    

    def listBuild(self,current,treeList,i) :

        if current == None :
            return

        else :
            treeList[i]=current



        self.listBuild(current.leftChild,treeList,self.left(i))
        self.listBuild(current.rightChild,treeList,self.right(i))


    def listRep(self,root) :

        h=self.height(root)+1

        treeList=[None for i in range(2**(h) - 1) ]

        self.listBuild(root,treeList,0)

        return treeList

    


    def left(self,i) :
        return 2*i+1

    def right(self,i) :
        return 2*i+2


    def printLevel(self,current,h) :

        if current == None :
            print("     ",end='')
            return

        if h > 0 :

            self.printLevel(current.leftChild,h-1)
            self.printLevel(current.rightChild,h-1)

        else:
            if current != None :
                print(current.featureDecision,end='    ')

            else :
                print("      ")

    def levelOrderDisp(self,root) :

        h=self.height(root)
        #print(h)
        #print(h)

        if h == 1 :
            h+=1

        spaceL=2**(h-2)

        spaceList=[spaceL]





        for i in range(h+1) :

            #print(i)

            spaceLTemp=spaceL
            spaceLTemp-=(2**i)
            spaceLTemp//=2
            #print(spaceLTemp)
            if i != 0 :
                print(abs(int(spaceLTemp))*' ',end='   ')

            else :
                print(abs(int(spaceLTemp))*' ',end='       ')

            spaceList.append(spaceLTemp)

            self.printLevel(root,i)
            print('')




def main():
    #print('Main called')
    tree = classTrees('dropped500redless.csv')

    tree.buildTree(None,tree.root,tree.trainingData)
    tree.predict()
    tree.accuracy()
    print()
    print("Preorder Traversal")
    print("-----------------------------------------------------------------------------------------")
    tree.preorderDisp(tree.root)
    print()
    print("Level order Traversal")
    print("-----------------------------------------------------------------------------------------")
    tree.levelOrderDisp(tree.root)
    print()
    print()
    treeList=tree.listRep(tree.root)

    '''for i in treeList :
        print(i.featureDecision)'''



#error detection in random sampling
if __name__==main():
    main()
    




