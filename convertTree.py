
#create a classs Node
class Node:
    def __init__(self,value,idnum):
        #initialize new node with left and right child as null
        self.left = None
        self.right = None
        self.value = value
        self.idnum = idnum

def isOperator(c):
    if(c == '&' or c == '|' or c == '!' or c == 'U' or c == 'G' or c == 'F' or c == 'R' or c == 'W' or c == 'M' or c == 'X'):
        return True
    else:
        return False


def constructTree(postfix):
    stack = []
    count = -1
    #traverse the postfix -> write the graph to a file?
    for char in postfix:
        count += 1
        #if operand then push it
        if(char.islower() or (char == 'N') or (char == 'T')):
            t = Node(char,count)
            stack.append(t)
        elif(char == '!'):
            t1 = stack.pop()
            t = Node(char,t1.idnum)
            #insert t1 to right of t
            t.right = t1
            #add the exp to stack
            stack.append(t)
            count -= 1
        #operator - unary then pop one and insert it to right
        elif(char == 'G' or char == 'X' or char == 'F'):
            t = Node(char,count)
            t1 = stack.pop()
            #insert t1 to right of t
            t.right = t1
            #add the exp to stack
            stack.append(t)
        #other operator - pop two nodes
        elif(isOperator(char)):
            t = Node(char,count)
            t1 = stack.pop()
            t2 = stack.pop()
            t.right = t1
            t.left = t2
            stack.append(t)
    #return the root
    t = stack.pop()
    return t



#convert to postfix
class Conversion:
    def __init__(self, capacity):
        self.top = -1
        self.capacity = capacity
        #stack array
        self.array = []
        self.output = []
        #precedence
        self.precedence = {'&':1, '|':1, 'M':2, 'R':2, 'W':2, 'U':2, '!':3, 'G':3, 'F':3, 'X':3}

    def isEmpty(self):
        return True if self.top == -1 else False

    def peek(self):
        return self.array[-1]

    def pop(self):
        if not self.isEmpty():
            self.top = self.top - 1
            return self.array.pop()
        else:
            return "$"

    def push(self, op):
        self.top = self.top + 1
        self.array.append(op)

    def isOperand(self, ch):
        return ch.islower() or (ch == 'N') or (ch == 'T')

    def notGreater(self, i): 
        try: 
            a = self.precedence[i] 
            b = self.precedence[self.peek()] 
            return True if a  <= b else False
        except KeyError:  
            return False
    def notUnary(self,i):
        j = self.peek()
        if((j == '!' or j == 'F' or j == 'G' or j == 'X') and (i == '!' or i == 'F' or i == 'G' or i == 'X')):
            return False
        return True

    def infixToPostfix(self, exp):
        # Iterate over the expression for conversion 
        for i in exp:
            if i == ' ':
                continue
            #print(i)
            if self.isOperand(i):#operand -> simple output
                self.output.append(i)
            elif(i == '('):#push opening bracket
                self.push(i)
            elif(i == ')'):#pop everything
                #(not self.isEmpty()) and
                while(self.peek() != '('):
                    a = self.pop()
                    self.output.append(a)
                #if(not self.isEmpty() and self.peek() != '('):
                    #return 1
                #else:
                    #self.pop()
                self.pop() #pop )
            #An operator is encountered -> push itr
            else:
                #while(not self.isEmpty() and self.notGreater(i) and self.notUnary(i) == True):
                    #self.output.append(self.pop())
                self.push(i)

        #pop
        while not self.isEmpty():
            self.output.append(self.pop())
        return "".join(self.output)

infixone = []
