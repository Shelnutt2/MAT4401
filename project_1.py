#Seth Shelnutt
#42941969
#MAT4401

#All code is released under the terms of the gpl v3 or later.
import argparse
import numpy as np
import sympy

parser = argparse.ArgumentParser(description='This is a program which demostrates various root finding and interpolation methods.')
parser.add_argument('--version', '-v', action='version', version='%(prog)s 0.8')
#parser.add_argument('Rows', metavar='Rows', type=eval, nargs='+', help='Rows of the matrix')


#initialize the arguments
args = parser.parse_args()

def midpoint():
   a = input('Please enter the lower bound of the interval: ')
   b = input('Please enter the upper bound of the interval: ')
   a = np.float(a)
   b = np.float(b)
   found = False
   i = 0
   fa = np.exp(-a/5) - np.sin(a)
   fb = np.exp(-b/5) - np.sin(b)
   c = 0.0
   fc = 0.0
   while(found == False):
      i = i+1
      c = (a+b)/2
      #print("x = " + str(c))
      fc = np.exp(-c/5) - np.sin(c)
      #print("y = " + str(fc))
      if abs(fc)<10**-7:
         print("Done")
         found = True
      elif fa*fc>0:
         a = c
         fa = np.exp(-a/5) - np.sin(a)
      elif fa*fc<0:
         b = c
         fb = np.exp((-b/5)) - np.sin(b)
   print("The zero is: " + str(c))
   print("It took " + str(i) + " interations.")
   return [c,i]

def newtons():
   a = input('Please enter a starting value: ')
   x = sympy.Symbol('x')
   y = sympy.exp((-x/5)) - sympy.sin(x)
   f= sympy.lambdify(x, y, 'numpy')
   yprime = y.diff(x)
   fprime = sympy.lambdify(x, yprime, 'numpy')
   found = False
   i = 0
   while(found == False):
      i = i+1
      xk = a - (f(a)/fprime(a))
      #print("x = " + str())
      #print("y = " + str(fc))
      if abs(f(xk))<10**-7:
         print("Done")
         found = True
      else:
         a = xk
   print("The zero is: " + str(xk))
   print("It took " + str(i) + " interations.")

def visual():
   print()
def interpolate():
   print()



def lagrange():
   print()

def root():
   print("Please choose one of the following options:")
   print("1. Visual Inspection")
   print("2. Bisection")
   print("3. Newtons")
   n = input("")
   rmenu[n]()

mmenu = {1:root,
         2:interpolate}
rmenu = {2:midpoint,
         3:newtons,
         1:visual}
imenu = {1:lagrange}

print("Please choose one of the following options:")
print("1. Root finding")
print("2. Interpolation")
print("")
n = input("")
mmenu[n]()


#print(midpoint(j,k))

