#Seth Shelnutt
#42941969
#MAT4401

#All code is released under the terms of the gpl v3 or later.
import argparse, math, sympy
import numpy as np
from matplotlib import pyplot
import easygui as eg

parser = argparse.ArgumentParser(description='This is a program which demostrates various root finding and interpolation methods.')
parser.add_argument('--version', '-v', action='version', version='%(prog)s 0.8')
#parser.add_argument('Rows', metavar='Rows', type=eval, nargs='+', help='Rows of the matrix')


#initialize the arguments
args = parser.parse_args()

def midpoint():
   a = input('Please enter the lower bound of the interval: ')
   b = input('Please enter the upper bound of the interval: ')
   print("Please click on the point you want to choose for intersection")
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
   def onclick(event):
    msg = "Do you want to choose x=%f"%(event.xdata)
    title = "Please Confirm"
    if eg.ynbox(msg, title, choices=('Yes', 'No'), image=None):     # show a Continue/Cancel dialog
       print 'x=%f'%(event.xdata)
       pyplot.close()  # user chose Continue
   
   a = input('Please enter the lower bound of the interval to graph: ')
   b = input('Please enter the upper bound of the interval to graph: ')
   a = np.float(a)
   b = np.float(b)
   x1 = np.arange(a,b,.01)
   y1 = np.sin(x1)
   y2 = np.exp(-x1/5)

   fig = pyplot.figure()
   ax = fig.add_subplot(111)

   ax.plot( x1,y1 )
   ax.plot( x1,y2 )

   ax = pyplot.axis([a, b, -1.5, 1.5])

   cid = fig.canvas.mpl_connect('button_press_event', onclick)


   pyplot.title( 'Plotting sin(x) and e^(-x/5)' )
   pyplot.xlabel( 'X Axis' )
   pyplot.ylabel( 'Y Axis' )
   pyplot.show()
       





def lagrange():
   n = input('What order lagrange polynomial do you want?: ')
   points=[]
   for i in np.arange(-5.,5.,(10./(n+1))):
      points.append([i,1./(1.+i**2.)])
   P = 0
   for i in points:
      ls = points[:]
      ls.remove(i)
      L  = 1
      for k in ls:
         x = sympy.Symbol('x')
         y = (x-k[0])/(i[0]-k[0])
         L = L * y
#      print f
      f2= sympy.simplify(L)
#      print f2
      P = P + (i[1]*L)
   Pn = sympy.simplify(P)
   print "The legrange Polynomail is: " + str(Pn)
   Pu = sympy.lambdify(x, P, 'numpy')
   Pe = i[1]- Pu(i[0])
   print "The error is: " +  str(Pe)
   #yprime = y.diff(x)
   #fprime = sympy.lambdify(x, yprime, 'numpy')

def piecewise():
   n = input('How many points do you want? ')
   points = []
   for i in np.arange(-5.,5.,(10./(n+1))):
      points.append([i,1./(1.+i**2.)])
   pw = []
   for k in points:
      if 0 == points.index(k):
         j = k
      else:
         x = sympy.Symbol('x')
         #print k
         #print j
         y = ((k[1]-j[1])/(k[0]-j[0]))*(x-j[0]) - j[1]
        # print y
         pw.append([y,j[0],k[0]])
         j = k
#   print pw
 #  cond = {}
  # for i in pw:
   #   cond[ =[(x>=i[2]) & (x<i[3])]
   #print cond
   #func = []
   #print func
   #for i in pw: func.append(sympy.lambdify(x, i[0], 'numpy'))
   #fA= np.piecewise(x,cond,func)
   #print fA
   print "the piecewise function is as follows: "
   print "F(x) = { "
   for i in pw:
      print str(i[0]) + "      for " + str(i[1]) + " < x < " + str(i[2]) 
   print "}"

def raised_cosine():
   n = input('How many points do you want? ')
   points = []
   for i in np.arange(-5.,5.,(10./(n+1))):
      points.append([i,1./(1.+i**2.)])
   pw = []
   j = points[0]
   x = sympy.Symbol('x')
   for k in points:
      if 0 == points.index(k):
         j = k
      else:
#         c = .5*(1+np.cosine(x))
         mu = j[0]+(x-j[0])/(k[0]-j[0])
#mu = x1+(x-x1)/(x2-x1)
         mu2 = (1-sympy.cos(mu*np.pi))/2;
         print(j[1]*(1-mu2)+k[1]*mu2)

#    def interpolate(a,b,x):
#        ''' cosine interpolation '''
#        curve = []
#        for i in xrange(x):
#            f = (1-cos(i/float(x)*pi))/2
#            curve.append((a*(1-f)+b*f))
#        return curve
   
def interpolate():
  while(True):
   print("Please choose one of the following options:")
   print("1. Lagrange")
   print("2. Piecewise")
   print("3. Raised Cosine")
   print("5. Quit")
   n = input("")
   if(imenu[n]()):
      break

def exit():
   return True

def root():
  while(True):
   print("Please choose one of the following options:")
   print("1. Visual Inspection")
   print("2. Bisection")
   print("3. Newtons")
   print("4. Quit")
   n = input("")
   if(rmenu[n]()):
      break

def main_menu():
  while(True):
   print("Please choose one of the following options:")
   print("1. Root finding")
   print("2. Interpolation")
   print("3. Quit")
   n = input("")
   if(mmenu[n]()):
      break

mmenu = {1:root,
         2:interpolate,
         3:exit,
         "q":exit}
rmenu = {2:midpoint,
         3:newtons,
         1:visual,
         'q':exit,
         4:exit}
imenu = {1:lagrange,
         2:piecewise,
         3:raised_cosine,
         5:exit,
         'q':exit}

main_menu()



