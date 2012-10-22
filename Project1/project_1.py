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

parser = argparse.ArgumentParser(description='This is a program which demostrates various root finding and interpolation methods.')
parser.add_argument('--version', '-v', action='version', version='%(prog)s 0.8')


#initialize the arguments
args = parser.parse_args()

def midpoint():
   while(True):
      a = input('Please enter the lower bound of the interval: ')
      b = input('Please enter the upper bound of the interval: ')
      a = np.float(a)
      b = np.float(b)
      found = False
      i = 0
      fa = np.exp(-a/5) - np.sin(a)
      fb = np.exp(-b/5) - np.sin(b)
      if fa*fb < 0:
         False
         break
      else:
         print "There is more than one or no sign change, please choose a different interval"
   c = 0.0
   fc = 0.0
   start = time.time()
   while(found == False):
      i = i+1
      c = (a+b)/2
      print("x = " + str(c))
      fc = np.exp(-c/5) - np.sin(c)
      print("y = " + str(fc))
      if abs(fc)<10**-7:
         print("Done")
         found = True
      elif fa*fc>0:
         a = c
         fa = np.exp(-a/5) - np.sin(a)
      elif fa*fc<0:
         b = c
         fb = np.exp((-b/5)) - np.sin(b)
   end = time.time()
   print("The zero is: " + str(c))
   print("It took " + str(i) + " interations and " + str(end-start) + " seconds.")
   return [c,i]

def newtons():
   a = input('Please enter a starting value: ')
   x = sympy.Symbol('x')
   start = time.time()
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
   end = time.time()
   print("The zero is: " + str(xk))
   print("It took " + str(i) + " interations and " + str(end-start) + " seconds.")

def secant():
   b = input('Please enter the first starting value: ')
   a = input('Please enter the second starting value: ')
   b = np.float(b)
   a = np.float(a)
   x = sympy.Symbol('x')
   start = time.time()
   y = sympy.exp((-x/5)) - sympy.sin(x)
   f = sympy.lambdify(x, y, 'numpy')
   found = False
   i = 0
   while(found == False):
      i = i+1
      xk = a - (f(a)*(a-b))/(f(a)-f(b))
      if abs(f(xk))<10**-7:
         print("Done")
         found = True
      else:
         b = a
         a = xk
   end = time.time()
   print("The zero is: " + str(xk))
   print("It took " + str(i) + " interations and " + str(end-start) + " seconds.")

def visual():
   def onclick(event):
    msg = "Do you want to choose x=%f"%(event.xdata)
    title = "Please Confirm"
    if eg.ynbox(msg, title, choices=('Yes', 'No'), image=None):     # show a Continue/Cancel dialog
       print 'x=%f'%(event.xdata)
       print 'The difference is: ' + str( np.sin(event.xdata) - np.exp(-event.xdata/5))
       pyplot.close()  # user chose Continue
   
   a = input('Please enter the lower bound of the interval to graph: ')
   b = input('Please enter the upper bound of the interval to graph: ')
   print("Please click on the point you want to choose for intersection")
   a = np.float(a)
   b = np.float(b)
   x1 = np.arange(a,b,.01)
   y1 = np.sin(x1)
   y2 = np.exp(-x1/5)
   fig = pyplot.figure()
   ax = fig.add_subplot(111)
   ax.plot( x1,y1 )
   ax.plot( x1,y2 )
   ax.axis([a, b, -1.5, 1.5])
   cid = fig.canvas.mpl_connect('button_press_event', onclick)
   pyplot.title( 'Plotting sin(x) and e^(-x/5)' )
   pyplot.xlabel( 'X Axis' )
   pyplot.ylabel( 'Y Axis' )
   pyplot.show()
       





def lagrange():
   n = input('What order lagrange polynomial do you want?: ')
   points=[]
   for i in np.arange(-5.,5.,(10./(n))):
      points.append([i,1./(1.+i**2.)])
   points.append([5.,1./(1.+5.**2.)])
   P = 0
   start = time.time()
   for i in points:
      ls = points[:]
      ls.remove(i)
      L  = 1
      for k in ls:
         x = sympy.Symbol('x')
         y = (x-k[0])/(i[0]-k[0])
         L = L * y
      f2= sympy.simplify(L)
      P = P + (i[1]*L)
   end = time.time()
   Pn = sympy.simplify(P)
   print "The lagrange Polynomail is: " + str(Pn)
   print "It took  " + str(end-start) + " seconds to calculate"
   Pu = sympy.lambdify(x, P, 'numpy')
   Pe = i[1]- Pu(i[0])
   print "The error is: " +  str(Pe)
   #yprime = y.diff(x)
   #fprime = sympy.lambdify(x, yprime, 'numpy')
   x = np.arange(-5.,5.,(10./(n)))
   xlot =  np.linspace(-5,5,1000)
   y = Pu(xlot)
   z = 1/(1+xlot**2)
   zpoint = 1/(1+x**2)
   pyplot.plot(xlot,y,label="Lagrange Polynomial")
   pyplot.plot(x,zpoint,'co')
   pyplot.plot(xlot,z,label="1/(1+x^2)")
   pyplot.legend()
   pyplot.ylim(0,2)
   pyplot.title(str(n)+"th Degree Lagrange Polynomial")
   pyplot.show()

def lform(i):
      ls = data[:]
      ls.remove(i)
      L = 1
      for k in ls:
         x = sympy.Symbol('x')
         y = (x-k[0])/(i[0]-k[0])
         L = L * y
      return(i[0]*L)

def lagrange2():
   n = input('What order lagrange polynomial do you want?: ')
   points=[]
   for i in np.arange(-5.,5.,(10./(n))):
      points.append([i,1./(1.+i**2.)])
   points.append([5.,1./(1.+5.**2.)])
   global data
   data = points[:]
   P = 0
   start = time.time()
   queue1 = Queue() #create a queue object
   procs = []
   pool = Pool(processes=24)              # start 24 worker processes
   result = pool.map(lform, points)
   x = sympy.Symbol('x')
   P = sum(result)
   end = time.time()
   Pn = sympy.simplify(P)
   print "The lagrange Polynomail is: " + str(Pn)
   print "It took  " + str(end-start) + " seconds to calculate"
   Pu = sympy.lambdify(x, P, 'numpy')
   Pe = points[len(points)-1][1]- Pu(points[len(points)-1][0])
   print "The error is: " +  str(Pe)
   #yprime = y.diff(x)
   #fprime = sympy.lambdify(x, yprime, 'numpy')
   x = np.arange(-5.,5.,(10./(n)))
   xlot =  np.linspace(-5,5,1000)
   y = Pu(xlot)
   z = 1/(1+xlot**2)
   zpoint = 1/(1+x**2)
   pyplot.plot(xlot,y,label="Lagrange Polynomial")
   pyplot.plot(x,zpoint,'co')
   pyplot.plot(xlot,z,label="1/(1+x^2)")
   pyplot.legend()
   pyplot.ylim(0,2)
   pyplot.title(str(n)+"th Degree Lagrange Polynomial")
   pyplot.show()

def piecewise():
   n = input('How many points do you want? ')
   points = []
   yvals=[]
   xvals=[]
   for i in np.arange(-5.,5.,(10./(n))):
      points.append([i,1./(1.+i**2.)])
      yvals.append(1./(1.+i**2.))
      xvals.append(i)
   yvals.append(1./(1.+5.**2.))
   xvals.append(5.)
   yvals = np.array(yvals)
   xvals = np.array(xvals)
   points.append([5.,1./(1.+5.**2.)])
   pw = []
   start = time.time()
   for k in points:
      if 0 == points.index(k):
         j = k
      else:
         xsy = sympy.Symbol('x')
         y = ((k[1]-j[1])/(k[0]-j[0]))*(xsy-j[0]) - j[1]
         pw.append([y,j[0],k[0]])
         j = k
   end = time.time()
   print "the piecewise function is as follows: "
   print "F(x) = { "
   for i in pw:
      print str(i[0]) + "      for " + str(i[1]) + " < x < " + str(i[2]) 
   print "}"
   print "It took  " + str(end-start) + " seconds to calculate"

   x = np.arange(-5.,5.,(10./(n)))
   xlot =  np.linspace(-5,5,1000)
   y = np.interp(xlot,xvals,yvals)
   z = 1/(1+xlot**2)
   zpoint = 1/(1+x**2)
   pyplot.plot(xlot,y,label="Linear Piecewise")
   pyplot.plot(x,zpoint,'co')
   pyplot.plot(xlot,z,label="1/(1+x^2)")
   pyplot.legend()
   pyplot.ylim(0,2)
   pyplot.title(str(n)+" point Linear Piecewise")
   pyplot.show()
   

def raised_cosine():
   n = input('How many points do you want? ')
   points = []
   for i in np.arange(-5.,5.,(10./(n))):
      points.append([i,1./(1.+i**2.)])
   points.append([5.,1./(1.+5.**2.)])
   pw = []
   j = points[0]
   x = sympy.Symbol('x')
   pw = []
   start = time.time()
   for k in points:
      if 0 == points.index(k):
         j = k
      else: 
         mu = j[0]+(x-j[0])/(k[0]-j[0])
         mu2 = (1-sympy.cos(mu*np.pi))/2
         y = (j[1]*(1-mu2)+k[1]*mu2)
         pw.append([y,j[0],k[0]])
         j = k
   end = time.time()
   for i in range(len(pw)):
    if i%2==0:
        print pw[i][0]
        W = sympy.lambdify(x,pw[i][0],'numpy')
        pw[i][0]  = -(pw[i][0].as_leading_term(x) - W(.5))*sympy.cos(np.pi*x)+W(.5)
        print pw[i][0]
   print "the piecewise function is as follows: "
   print "F(x) = { "
   for i in pw:
      print str(i[0]) + "      for " + str(i[1]) + " < x < " + str(i[2]) 
   print "}"
   print "It took  " + str(end-start) + " seconds to calculate"
   
   x = sympy.Symbol('x')
   for i in pw:
      y = sympy.lambdify(x,i[0],'numpy')
      xlot =  np.linspace(i[1],i[2],1000)
      pyplot.plot(xlot,y(xlot))
   xlot2 =  np.linspace(-5,5,1000)
   z = 1/(1+xlot2**2)
   pyplot.plot(xlot2,z,label="1/(1+x^2)")
   pyplot.legend()
   pyplot.ylim(0,2)
   pyplot.title(str(n)+" point Linear Piecewise")
   pyplot.show()

def lsa():
   n = input('What order least square approximation polynomial do you want?: ')
   points=[]
   yvals=[]
   xvals=[]
   start = time.time()
   for i in np.arange(-5.,5.,(10./(n))):
      points.append([i,1./(1.+i**2.)])
      yvals.append(1./(1.+i**2.))
      xvals.append(i)
   points.append([5.,1./(1.+5.**2.)])
   yvals.append(1./(1.+5.**2.))
   xvals.append(5.)
   yvals = np.array(yvals)
   xvals = np.array(xvals)
   A = np.vander(xvals, n)
   # find the x that minimizes the norm of Ax-y
   (coeffs, residuals, rank, sing_vals) = np.linalg.lstsq(A, yvals)
   # create a polynomial using coefficients
   f = np.poly1d(coeffs)
   end = time.time()
   print "The least square approximation is: " + str(f)
   print "It took  " + str(end-start) + " seconds to calculate"
   ss_err=(residuals**2).sum()
   ss_tot=((yvals-yvals.mean())**2).sum()
   rsquared=1-(ss_err/ss_tot)
   print("The rsquared value is: " + str(rsquared))
   x = np.arange(-5.,5.,(10./(n)))
   xlot =  np.linspace(-5,5,1000)
   y = f(xlot)
   z = 1/(1+xlot**2)
   zpoint = 1/(1+x**2)
   pyplot.plot(xlot,y,label="Least Square Polynomial")
   pyplot.plot(x,zpoint,'co')
   pyplot.plot(xlot,z,label="1/(1+x^2)")
   pyplot.legend()
   pyplot.ylim(0,2)
   pyplot.title(str(n)+"th Degree Least Squares Polynomial")
   pyplot.show()

def interpolate():
  while(True):
   print("Please choose one of the following options:")
   print("1. Lagrange")
   print("2. Piecewise")
   print("3. Raised Cosine")
   print("4. Least Square Approximation")
   print("5. Lagrange Optimized")
   print("6. Quit")
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
   print("4. Secant (Newton Optimized)")
   print("5. Quit")
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
         4:secant,
         'q':exit,
         5:exit}
imenu = {1:lagrange,
         2:piecewise,
         3:raised_cosine,
         4:lsa,
         5:lagrange2,
         6:exit,
         'q':exit}
main_menu()

