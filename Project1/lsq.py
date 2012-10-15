import argparse, math, sympy, time
import numpy as np
from matplotlib import pyplot
import easygui as eg

def lsa(n):
   
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
   return f

def lagrange(n):
#   n = input('What order lagrange polynomial do you want?: ')
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
#      print f
      f2= sympy.simplify(L)
#      print f2
      P = P + (i[1]*L)
   end = time.time()
   Pn = sympy.simplify(P)
   print "The legrange Polynomail is: " + str(Pn)
   print "It took  " + str(end-start) + " seconds to calculate"
   Pu = sympy.lambdify(x, P, 'numpy')
   Pe = i[1]- Pu(i[0])
   print "The error is: " +  str(Pe)
   return Pu

def piecewise(n):
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
   xlot =  np.linspace(-5,5,1000)
   y = np.interp(xlot,xvals,yvals)
   return y


xfive = np.arange(-5.,5.,(10./(5)))
xten = np.arange(-5.,5.,(10./(10)))
xtwenty = np.arange(-5.,5.,(10./(20)))

xlot =  np.linspace(-5,5,1000)
yfive = lsa(5)(xlot)
yten = lsa(10)(xlot)
ytwenty = lsa(20)(xlot)
z = 1/(1+xlot**2)
zpoint = 1/(1+xtwenty**2)

pyplot.plot(xlot,yfive,label="5th degree Least Square Polynomial")
pyplot.plot(xlot,yten,label="10th degree Least Square Polynomial")
pyplot.plot(xlot,ytwenty,label="20th degree Least Square Polynomial")
pyplot.plot(xtwenty,zpoint,'co')
pyplot.plot(xlot,z,label="1/(1+x^2)")
pyplot.legend()
pyplot.ylim(0,2)
pyplot.title("5th, 10th and 20th Least Squares Polynomial")
pyplot.show()

yfive = lagrange(5)(xlot)
yten = lagrange(10)(xlot)
ytwenty = lagrange(20)(xlot)
z = 1/(1+xlot**2)
zpoint = 1/(1+xtwenty**2)

pyplot.plot(xlot,yfive,label="5th degree Least Square Polynomial")
pyplot.plot(xlot,yten,label="10th degree Least Square Polynomial")
pyplot.plot(xlot,ytwenty,label="20th degree Least Square Polynomial")
pyplot.plot(xtwenty,zpoint,'co')
pyplot.plot(xlot,z,label="1/(1+x^2)")
pyplot.legend()
pyplot.ylim(0,2)
pyplot.title("5th, 10th and 20th Least Squares Polynomial")
pyplot.show()

yfive = piecewise(5)
yten = piecewise(10)
ytwenty = piecewise(20)
z = 1/(1+xlot**2)
zpoint = 1/(1+xtwenty**2)

pyplot.plot(xlot,yfive,label="5 point Linear Piecewise")
pyplot.plot(xlot,yten,label="10 point Linear Piecewise")
pyplot.plot(xlot,ytwenty,label="20 point Linear Piecewise")
pyplot.plot(xtwenty,zpoint,'co')
pyplot.plot(xlot,z,label="1/(1+x^2)")
pyplot.legend()
pyplot.ylim(0,2)
pyplot.title("5, 10 and 20 Point Linear Piecewise")
pyplot.show()
