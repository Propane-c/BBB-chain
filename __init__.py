import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath)

if __name__ == '__main__':
    print(sys.path) 