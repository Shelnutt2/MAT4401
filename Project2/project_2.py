#!/bin/python
#Seth Shelnutt
#42941969
#MAT4401

#All code is released under the terms of the gpl v3 or later.
import argparse, math, sympy, time
from multiprocessing import Process, Queue, Pool
import numpy as np
from matplotlib import pyplot
from sympy.utilities.lambdify import lambdastr
import easygui as eg

parser = argparse.ArgumentParser(description='This is a program which demostrates various numerical integration and ode methods.')
parser.add_argument('--version', '-v', action='version', version='%(prog)s 1.0')


#initialize the arguments
args = parser.parse_args()

def riemann():
   x1 = 0 #Set intial x values
   x2 = 4 #Set intial x values
   h = input('Please enter the step size: ') #Get step size from user
   i = int(abs ((x2-x1)/h))
   n=0   # initialize
   A= 0.0 # initialize
   x0 = x1 # initialize
   x = sympy.Symbol('x') #Make x symbolic
   y1 = sympy.exp(3*x) #make f(x)
   y2 = 1 + sympy.sin(10*np.pi*x) #make f(x)
   flist = [y1,y2] # list of functions to iterate over
   for func in flist: # iterate over list
      S= 0.0
      f = sympy.lambdify(x, func, 'numpy') #Make it a function
      start = time.time()
      for n in range(i):    # Begin Numerical Integration
         A_prime = f(x0) * h #Area is y value * step size
         x0 += h # Move to next X
         S += A_prime  #Add to current sum
      end = time.time()
      print("Area Under the Curve "+str(func)+" ="+ str(S)+"\nIt took "+str(end-start)+" seconds to calulate.")

 

def trapezoidINT():
   x1 = 0 #Set intial x values
   x2 = 4 #Set intial x values
   h = input('Please enter the step size: ') #Get step size from user
   n = (int)((x2-x1)/h)
   x = sympy.Symbol('x') #Make x symbolic
   y1 = sympy.exp(3*x) #make f(x)
   y2 = 1 + sympy.sin(10*np.pi*x) #make f(x)
   flist = [y1,y2] # list of functions to iterate over
   for func in flist: # iterate over list
      f = sympy.lambdify(x, func, 'numpy') #Make it a function
      start = time.time()
      S = 0
      for xk in xrange(0, n): 
         S +=f(x1+xk*h) #Summation of y values
      A = (h/2 *(f(x1)+f(x2)))+(h*S) #Tapezoid area formula
      end = time.time()
      print("Area Under the Curve "+str(func)+" ="+ str(A)+"\nIt took "+str(end-start)+" seconds to calulate.")

def simpson():
   x1 = 0 #Set intial x values
   x2 = 4 #Set intial x values
   h = input('Please enter the step size: ') #Get step size from user
   n = (int)((x2-x1)/h) #Set number of steps
   x = sympy.Symbol('x') #Make x symbolic
   y1 = sympy.exp(3*x) #make f(x)
   y2 = 1 + sympy.sin(10*np.pi*x) #make f(x)
   flist = [y1,y2] # list of functions to iterate over
   for func in flist: # iterate over list
      f = sympy.lambdify(x, func, 'numpy') #Make it a function
      start = time.time()
      S = f(x1) #Get intial y value
      for i in range(1, n, 2): #Iterate from 1 to n, using only odd numers
         S += 4 * f(x1 + h * i)
      for i in range(2, n-1, 2): #Iterate from 2 to n-1, using only odd numers
         S += 2 * f(x1 + h * i)
      S += f(x2) #Add final y value
      A = h * S / 3 #Set area to equal step size times 1/3 of the summation
      end = time.time()
      print("Area Under the Curve "+str(func)+" ="+ str(A)+"\nIt took "+str(end-start)+" seconds to calulate.")


def euler():
   h = input('Please enter the step size: ') #Get step size from user
   y = sympy.Symbol('y') #Make y symbolic
   x = sympy.Symbol('x') #Make x symbolic
   y1 = 3*y #make f(x)
   y2 = 1/(1+x**2)-2*y**2 #make f(x)
   f2sol = sympy.lambdify(x, x/(1+x**2), 'numpy') #Make it a function 
   f1sol = sympy.lambdify(x, sympy.exp(3*x), 'numpy') #Make it a function
   flist = [[y1,0,3,1,f1sol,0,5000],[y2,0,10,0,f2sol,0,.8]] #List to iterate over with initial values and solutions to graph
   for func in flist: #Iterate
      xlot =  np.linspace(func[1],func[2],1000) #x values for graphing the solution
      n = (int)((func[2]-func[1])/h) #Number of steps
      xk = func[1] #Set initial Values
      ya = func[3] #Set initial Values
      xpoints = []
      ypoints = []
      xpoints.append(xk) #Add inital points to point list for graphing
      ypoints.append(ya) #Add inital points to point list for graphing
      f = sympy.lambdify((x, y), func[0], 'numpy') #Make it a function 
      start = time.time() 
      for i in range(n):
          ya += h * f(xk, ya) #Euler Method
          xk += h #Move to next x values
          xpoints.append(xk) #Add x coordinate to x list for graph
          ypoints.append(ya) #Add y coordinate to y list for graph
      end = time.time()
      print("It took " + str(end-start)+" seconds to calculate")
      print("The maximum error is: " + str(abs(func[4](xk)-ya)))
      pyplot.plot(xpoints,ypoints,label="Euler")
      Label = "e^3x"
      pyplot.title("y'=3y Euler Plot")
      if func[2] == 10:
         Label = "x/(1+x^2)"
         pyplot.title("y'=1/(1+x^2)-2y^2 Euler Plot")
      pyplot.plot(xlot,func[4](xlot),label=Label)#Graph
      pyplot.legend()
      pyplot.ylim(func[5],func[6])
      pyplot.show()

def midpoint(): #Define midpoint method
   h = input('Please enter the step size: ') #Get step size from user
   y = sympy.Symbol('y') #Make y symbolic
   x = sympy.Symbol('x') #Make x symbolic
   y1 = 3*y #make f(x)
   y2 = 1/(1+x**2)-2*y**2 #make f(x)
   f2sol = sympy.lambdify(x, x/(1+x**2), 'numpy') #Make it a function 
   f1sol = sympy.lambdify(x, sympy.exp(3*x), 'numpy') #Make it a function
   flist = [[y1,0,3,1,f1sol,0,5000],[y2,0,10,0,f2sol,0,.8]] #List to iterate over with initial values and solutions to graph
   for func in flist: #Iterate
      xlot =  np.linspace(func[1],func[2],1000) #x values for graphing the solution
      n = (int)((func[2]-func[1])/h)
      xk = func[1] #Set initial Values
      ya = func[3] #Set initial Values
      xpoints = []
      ypoints = []
      xpoints.append(xk) #Add inital points to point list for graphing
      ypoints.append(ya) #Add inital points to point list for graphing
      f = sympy.lambdify((x, y), func[0], 'numpy') #Make it a function  
      start = time.time()
      for i in range(n): #Iterate over the nubmer of steps
          ya += h*f(xk + 0.5*h, ya + 0.5*h*f(xk,ya)) #Midpoint method
          xk += h #Move to next x values
          xpoints.append(xk) #Add x coordinate to x list for graph
          ypoints.append(ya) #Add y coordinate to y list for graph
      end = time.time()
      print("It took " + str(end-start)+" seconds to calculate")
      print("The maximum error is: " + str(abs(func[4](xk)-ya)))
      pyplot.plot(xpoints,ypoints,label="Midpoint")
      Label = "e^3x"
      pyplot.title("y'=3y Midpoint Plot")
      if func[2] == 10:
         Label = "x/(1+x^2)"
         pyplot.title("y'=1/(1+x^2)-2y^2 Midpoint Plot")
      pyplot.plot(xlot,func[4](xlot),label=Label) #Graph
      pyplot.legend()
      pyplot.ylim(func[5],func[6])
      pyplot.show()

def trapezoidODE():
   h = input('Please enter the step size: ') #Get step size from user
   y = sympy.Symbol('y') #Make y symbolic
   x = sympy.Symbol('x') #Make x symbolic
   y1 = 3*y #make f(x)
   y2 = 1/(1+x**2)-2*y**2 #make f(x)
   f2sol = sympy.lambdify(x, x/(1+x**2), 'numpy') #Make it a function 
   f1sol = sympy.lambdify(x, sympy.exp(3*x), 'numpy') #Make it a function
   flist = [[y1,0,3,1,f1sol,0,5000],[y2,0,10,0,f2sol,0,.8]] #List to iterate over with initial values and solutions to graph
   for func in flist: #Iterate
      xlot =  np.linspace(func[1],func[2],1000) #x values for graphing the solution
      n = (int)((func[2]-func[1])/h)
      xk = func[1] #Set initial Values
      ya = func[3] #Set initial Values
      xpoints = []
      ypoints = []
      xpoints.append(xk) #Add inital points to point list for graphing
      ypoints.append(ya) #Add inital points to point list for graphing
      f = sympy.lambdify((x, y), func[0], 'numpy') #Make it a function  
      start = time.time()
      for i in range(n):
          ya += .5*h*f(xk , ya) + .5*h*f(xk+h,ya+h*f(xk,ya)) #Trapezoid iterative method
          xk += h #Move to next x values
          xpoints.append(xk) #Add x coordinate to x list for graph
          ypoints.append(ya) #Add y coordinate to y list for graph
      end = time.time()
      print("It took " + str(end-start)+" seconds to calculate")
      print("The maximum error is: " + str(abs(func[4](xk)-ya)))
      pyplot.plot(xpoints,ypoints,label="Trapezoid")
      Label = "e^3x"
      pyplot.title("y'=3y Trapezoid Plot")
      if func[2] == 10:
         Label = "x/(1+x^2)"
         pyplot.title("y'=1/(1+x^2)-2y^2 Trapezoid Plot")
      pyplot.plot(xlot,func[4](xlot),label=Label)#Graph
      pyplot.legend()
      pyplot.ylim(func[5],func[6])
      pyplot.show()


def NumODE(): #Define Interpolation menu
  while(True):
   print("Please choose one of the following options:")
   print("1. Eulers Method")
   print("2. Midpoint Method")
   print("3. Trapezoid Method")
   print("4. Quit")
   n = input("")
   if(omenu[n]()):
      break

def exit(): #Define Exit
   return True

def NumInt(): #Define root finding menu
  while(True):
   print("Please choose one of the following options:")
   print("1. Riemann Sums")
   print("2. Trapezoid Rule")
   print("3. Simpsons Rule")
   print("4. Quit")
   n = input("")
   if(imenu[n]()):
      break

def main_menu(): #Define main menu
  while(True):
   print("Please choose one of the following options:")
   print("1. Numerical Integration")
   print("2. Numerical ODE Solving Routines")
   print("3. Quit")
   n = input("")
   if(mmenu[n]()):
      break
#Define dictionaries of menus
mmenu = {1:NumInt,
         2:NumODE,
         3:exit,
         "q":exit}
imenu = {1:riemann,
         2:trapezoidINT,
         3:simpson,
         'q':exit,
         4:exit}
omenu = {1:euler,
         2:midpoint,
         3:trapezoidODE,
         4:exit,
         'q':exit}
main_menu() #run main menu

