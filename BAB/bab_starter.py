import picos as pic
from picos import RealVariable
from copy import deepcopy
from heapq import *
import heapq as hq
import numpy as np
import itertools
import math
counter = itertools.count() 

class BBTreeNode():
    def __init__(self, vars = [], constraints = [], objective='', prob=None):
        self.vars = vars
        self.constraints = constraints
        self.objective = objective
        self.prob = prob

    def __deepcopy__(self, memo):
        '''
        Deepcopies the picos problem
        This overrides the system's deepcopy method bc it doesn't work on classes by itself
        '''
        newprob = pic.Problem.clone(self.prob)
        return BBTreeNode(self.vars, newprob.constraints, self.objective, newprob)
    
    def buildProblem(self):
        '''
        Bulids the initial Picos problem
        '''
        prob=pic.Problem()

        if type(self.constraints) is not list:
           c = [self.constraints[x] for x in self.constraints]
        else:
            c = self.constraints
        prob.add_list_of_constraints(c)    
        
        prob.set_objective('max', self.objective)
        self.prob = prob
        return self.prob

    def is_integral(self):
        '''
        Checks if all variables (excluding the one we're maxing) are integers
        '''
        for v in self.vars[:-1]:
            if v.value == None or abs(round(v.value) - float(v.value)) > 1e-4 :
                return False
        return True

    def branch_floor(self, branch_var):
        '''
        Makes a child where xi <= floor(xi)
        '''
        n1 = deepcopy(self)
        n1.prob.add_constraint( branch_var <= math.floor(branch_var.value) ) # add in the new binary constraint
        n1.constraints = n1.prob.constraints

        return n1

    def branch_ceil(self, branch_var):
        '''
        Makes a child where xi >= ceiling(xi)
        '''
        n2 = deepcopy(self)
        n2.prob.add_constraint( branch_var >= math.ceil(branch_var.value) ) # add in the new binary constraint
        n2.constraints = n2.prob.constraints

        return n2


    def bbsolve(self):
        '''
        Use the branch and bound method to solve an integer program
        This function should return:
            return bestres, bestnode_vars

        where bestres = value of the maximized objective function
              bestnode_vars = the list of variables that create bestres
        '''

        # these lines build up the initial problem and adds it to a heap
        root = self
        bestres = -1e20 # a small arbitrary initial best objective value
        bestnode_vars = root.vars # initialize bestnode_vars to the root vars

        try:
            res = root.buildProblem().solve(solver='cvxopt')
            heap = [(res, next(counter), root)]
        except pic.modeling.problem.SolutionFailure:
            # Base case: no valid solution
            return bestres, list(bestnode_vars)

        # Base case: is integral
        if self.is_integral():
            return float(res.reportedValue), list(self.vars)

        # Recursive case
        children = []
        for v in self.vars[:-1]:
            if abs(round(v.value) - float(v.value)) > 1e-4: # v isn't an integer yet
                children.append(self.branch_floor(v))
                children.append(self.branch_ceil(v))

        def get_val(x):
            out = x.bbsolve()[0]
            try:
                return float(out.reportedValue)
            except:
                return out

        opt = max(children, key = get_val)
        return get_val(opt), list(opt.vars)
 
