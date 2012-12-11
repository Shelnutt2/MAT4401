#!/bin/python
#Seth Shelnutt
#42941969
#MAT4401

import argparse, math, sympy, time
import numpy as np
from matplotlib import pyplot

def Euler(h,f,x1,x2,y1):
      y = sympy.Symbol('y') #Make y symbolic
      x = sympy.Symbol('x') #Make x symbolic
      func = f
      n = (int)((x2-x1)/h) #Number of steps
      xk = x1 #Set initial Values
      ya = y1 #Set initial Values
      xpoints = []
      ypoints = []
      xpoints.append(xk) #Add inital points to point list for graphing
      ypoints.append(ya) #Add inital points to point list for graphing

      start = time.time() 
      for i in range(n):
          ya += h * f(xk, ya) #Euler Method
          xk += h #Move to next x values
          xpoints.append(xk) #Add x coordinate to x list for graph
          ypoints.append(ya) #Add y coordinate to y list for graph
      end = time.time()
      print("It took " + str(end-start)+" seconds to calculate")
      return[xpoints,ypoints]

def midpoint(h,f,x1,x2,y1): #Define midpoint method
      y = sympy.Symbol('y') #Make y symbolic
      x = sympy.Symbol('x') #Make x symbolic
      n = (int)((x2-x1)/h)
      xk = x1 #Set initial Values
      ya = y1 #Set initial Values
      xpoints = []
      ypoints = []
      xpoints.append(xk) #Add inital points to point list for graphing
      ypoints.append(ya) #Add inital points to point list for graphing

      start = time.time()
      for i in range(n): #Iterate over the nubmer of steps
          ya += h*f(xk + 0.5*h, ya + 0.5*h*f(xk,ya)) #Midpoint method
          xk += h #Move to next x values
          xpoints.append(xk) #Add x coordinate to x list for graph
          ypoints.append(ya) #Add y coordinate to y list for graph
      end = time.time()
      print("It took " + str(end-start)+" seconds to calculate")
      return[xpoints,ypoints]

def trapezoidODE(h,f,x1,x2,y1):
      y = sympy.Symbol('y') #Make y symbolic
      x = sympy.Symbol('x') #Make x symbolic
      n = (int)((x2-x1)/h)
      xk = x1 #Set initial Values
      ya = y1 #Set initial Values
      xpoints = []
      ypoints = []
      xpoints.append(xk) #Add inital points to point list for graphing
      ypoints.append(ya) #Add inital points to point list for graphing
      start = time.time()
      for i in range(n):
          ya += .5*h*f(xk , ya) + .5*h*f(xk+h,ya+h*f(xk,ya)) #Trapezoid iterative method
          xk += h #Move to next x values
          xpoints.append(xk) #Add x coordinate to x list for graph
          ypoints.append(ya) #Add y coordinate to y list for graph
      end = time.time()
      print("It took " + str(end-start)+" seconds to calculate")
      return[xpoints,ypoints]


xlot =  np.linspace(0,3,1000)

y = sympy.Symbol('y') #Make y symbolic
x = sympy.Symbol('x') #Make x symbolic
y1 = 3*y 
y2 = 1/(1+x**2)-2*y**2
f = sympy.lambdify((x, y), y1, 'numpy') #Make it a function
yone = Euler(.1,f,0,3,1)
yfive = Euler(.05,f,0,3,1)
yoone = Euler(.01,f,0,3,1)
yofive = Euler(.005,f,0,3,1)
z = np.exp(3*xlot)

pyplot.plot(yone[0],yone[1],label="Step .1")
pyplot.plot(yfive[0],yfive[1],label="Step .05")
pyplot.plot(yoone[0],yoone[1],label="Step .01")
pyplot.plot(yofive[0],yofive[1],label="Step .005")
pyplot.plot(xlot,z,label="e^3x")
pyplot.legend(loc=2)
pyplot.ylim(0,5000)
pyplot.title("Euler method")
pyplot.show()

f = sympy.lambdify((x, y), y2, 'numpy') #Make it a function
yone = Euler(.1,f,0,10,0)
yfive = Euler(.05,f,0,10,0)
yoone = Euler(.01,f,0,10,0)
yofive = Euler(.005,f,0,10,0)
z = xlot/(1+xlot**2)

pyplot.plot(yone[0],yone[1],label="Step .1")
pyplot.plot(yfive[0],yfive[1],label="Step .05")
pyplot.plot(yoone[0],yoone[1],label="Step .01")
pyplot.plot(yofive[0],yofive[1],label="Step .005")
pyplot.plot(xlot,z,label="1/(1+x^2)")
pyplot.legend()
pyplot.ylim(0,0.8)
pyplot.title("Euler method")
pyplot.show()

f = sympy.lambdify((x, y), y1, 'numpy') #Make it a function
yone = midpoint(.1,f,0,3,1)
yfive = midpoint(.05,f,0,3,1)
yoone = midpoint(.01,f,0,3,1)
yofive = midpoint(.005,f,0,3,1)
z = np.exp(3*xlot)

pyplot.plot(yone[0],yone[1],label="Step .1")
pyplot.plot(yfive[0],yfive[1],label="Step .05")
pyplot.plot(yoone[0],yoone[1],label="Step .01")
pyplot.plot(yofive[0],yofive[1],label="Step .005")
pyplot.plot(xlot,z,label="e^3x")
pyplot.legend(loc=2)
pyplot.ylim(0,5000)
pyplot.title("Midpoint method")
pyplot.show()

f = sympy.lambdify((x, y), y2, 'numpy') #Make it a function
yone = midpoint(.1,f,0,10,0)
yfive = midpoint(.05,f,0,10,0)
yoone = midpoint(.01,f,0,10,0)
yofive = midpoint(.005,f,0,10,0)
z = xlot/(1+xlot**2)

pyplot.plot(yone[0],yone[1],label="Step .1")
pyplot.plot(yfive[0],yfive[1],label="Step .05")
pyplot.plot(yoone[0],yoone[1],label="Step .01")
pyplot.plot(yofive[0],yofive[1],label="Step .005")
pyplot.plot(xlot,z,label="1/(1+x^2)")
pyplot.legend()
pyplot.ylim(0,0.8)
pyplot.title("Midpoint method")
pyplot.show()

f = sympy.lambdify((x, y), y1, 'numpy') #Make it a function
yone = trapezoidODE(.1,f,0,3,1)
yfive = trapezoidODE(.05,f,0,3,1)
yoone = trapezoidODE(.01,f,0,3,1)
yofive = trapezoidODE(.005,f,0,3,1)
z = np.exp(3*xlot)

pyplot.plot(yone[0],yone[1],label="Step .1")
pyplot.plot(yfive[0],yfive[1],label="Step .05")
pyplot.plot(yoone[0],yoone[1],label="Step .01")
pyplot.plot(yofive[0],yofive[1],label="Step .005")
pyplot.plot(xlot,z,label="e^3x")
pyplot.legend(loc=2)
pyplot.ylim(0,5000)
pyplot.title("Trapezoid method")
pyplot.show()

f = sympy.lambdify((x, y), y2, 'numpy') #Make it a function
yone = trapezoidODE(.1,f,0,10,0)
yfive = trapezoidODE(.05,f,0,10,0)
yoone = trapezoidODE(.01,f,0,10,0)
yofive = trapezoidODE(.005,f,0,10,0)
z = xlot/(1+xlot**2)

pyplot.plot(yone[0],yone[1],label="Step .1")
pyplot.plot(yfive[0],yfive[1],label="Step .05")
pyplot.plot(yoone[0],yoone[1],label="Step .01")
pyplot.plot(yofive[0],yofive[1],label="Step .005")
pyplot.plot(xlot,z,label="1/(1+x^2)")
pyplot.legend()
pyplot.ylim(0,0.8)
pyplot.title("Trapezoid method")
pyplot.show()

f = sympy.lambdify((x, y), y1, 'numpy') #Make it a function
yoneE = Euler(.1,f,0,3,1)
yonem = midpoint(.1,f,0,3,1)
yonet = trapezoidODE(.1,f,0,3,1)
z = np.exp(3*xlot)

pyplot.plot(yoneE[0],yoneE[1],label="Euler")
pyplot.plot(yonem[0],yonem[1],label="Midpoint")
pyplot.plot(yonet[0],yonet[1],label="Trapezoid")
pyplot.plot(xlot,z,label="e^3x")
pyplot.legend(loc=2)
pyplot.ylim(0,5000)
pyplot.title("Step size .1")
pyplot.show()

f = sympy.lambdify((x, y), y1, 'numpy') #Make it a function
yoneE = Euler(.05,f,0,3,1)
yonem = midpoint(.05,f,0,3,1)
yonet = trapezoidODE(.05,f,0,3,1)
z = np.exp(3*xlot)

pyplot.plot(yoneE[0],yoneE[1],label="Euler")
pyplot.plot(yonem[0],yonem[1],label="Midpoint")
pyplot.plot(yonet[0],yonet[1],label="Trapezoid")
pyplot.plot(xlot,z,label="e^3x")
pyplot.legend(loc=2)
pyplot.ylim(0,5000)
pyplot.title("Step size .05")
pyplot.show()

f = sympy.lambdify((x, y), y1, 'numpy') #Make it a function
yoneE = Euler(.01,f,0,3,1)
yonem = midpoint(.01,f,0,3,1)
yonet = trapezoidODE(.01,f,0,3,1)
z = np.exp(3*xlot)

pyplot.plot(yoneE[0],yoneE[1],label="Euler")
pyplot.plot(yonem[0],yonem[1],label="Midpoint")
pyplot.plot(yonet[0],yonet[1],label="Trapezoid")
pyplot.plot(xlot,z,label="e^3x")
pyplot.legend(loc=2)
pyplot.ylim(0,5000)
pyplot.title("Step size .01")
pyplot.show()

f = sympy.lambdify((x, y), y1, 'numpy') #Make it a function
yoneE = Euler(.005,f,0,3,1)
yonem = midpoint(.005,f,0,3,1)
yonet = trapezoidODE(.005,f,0,3,1)
z = np.exp(3*xlot)

pyplot.plot(yoneE[0],yoneE[1],label="Euler")
pyplot.plot(yonem[0],yonem[1],label="Midpoint")
pyplot.plot(yonet[0],yonet[1],label="Trapezoid")
pyplot.plot(xlot,z,label="e^3x")
pyplot.legend(loc=2)
pyplot.ylim(0,5000)
pyplot.title("Step size .005")
pyplot.show()

f = sympy.lambdify((x, y), y2, 'numpy') #Make it a function
yoneE = Euler(.1,f,0,10,0)
yonem = midpoint(.1,f,0,10,0)
yonet = trapezoidODE(.1,f,0,10,0)
z = xlot/(1+xlot**2)

pyplot.plot(yoneE[0],yoneE[1],label="Euler")
pyplot.plot(yonem[0],yonem[1],label="Midpoint")
pyplot.plot(yonet[0],yonet[1],label="Trapezoid")
pyplot.plot(xlot,z,label="1/(1+x^2)")
pyplot.legend()
pyplot.ylim(0,0.8)
pyplot.title("Step size .1")
pyplot.show()

f = sympy.lambdify((x, y), y2, 'numpy') #Make it a function
yoneE = Euler(.05,f,0,10,0)
yonem = midpoint(.05,f,0,10,0)
yonet = trapezoidODE(.05,f,0,10,0)
z = xlot/(1+xlot**2)

pyplot.plot(yoneE[0],yoneE[1],label="Euler")
pyplot.plot(yonem[0],yonem[1],label="Midpoint")
pyplot.plot(yonet[0],yonet[1],label="Trapezoid")
pyplot.plot(xlot,z,label="1/(1+x^2)")
pyplot.legend()
pyplot.ylim(0,0.8)
pyplot.title("Step size .05")
pyplot.show()

f = sympy.lambdify((x, y), y2, 'numpy') #Make it a function
yoneE = Euler(.01,f,0,10,0)
yonem = midpoint(.01,f,0,10,0)
yonet = trapezoidODE(.01,f,0,10,0)
z = xlot/(1+xlot**2)

pyplot.plot(yoneE[0],yoneE[1],label="Euler")
pyplot.plot(yonem[0],yonem[1],label="Midpoint")
pyplot.plot(yonet[0],yonet[1],label="Trapezoid")
pyplot.plot(xlot,z,label="1/(1+x^2)")
pyplot.legend()
pyplot.ylim(0,0.8)
pyplot.title("Step size .01")
pyplot.show()

f = sympy.lambdify((x, y), y2, 'numpy') #Make it a function
yoneE = Euler(.005,f,0,10,0)
yonem = midpoint(.005,f,0,10,0)
yonet = trapezoidODE(.005,f,0,10,0)
z = xlot/(1+xlot**2)

pyplot.plot(yoneE[0],yoneE[1],label="Euler")
pyplot.plot(yonem[0],yonem[1],label="Midpoint")
pyplot.plot(yonet[0],yonet[1],label="Trapezoid")
pyplot.plot(xlot,z,label="1/(1+x^2)")
pyplot.legend()
pyplot.ylim(0,0.8)
pyplot.title("Step size .005")
pyplot.show()
