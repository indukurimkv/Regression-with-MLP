import numpy as np

class Node:
    lookup = {}
    @staticmethod
    def register(key,val):
        Node.lookup[key] = val
    def __init__(self, func = lambda a,b: a+b, ids=None):
        self.func = func
        self.children = ids
        
    def evaluate(self):
        args = []
        for child in self.children:
            if not isinstance(child, Node):
                args.append(Node.lookup[child])
            else:
                args.append(child.evaluate())
                
        return self.func(*args)
    

if __name__ == "__main__":
    pass