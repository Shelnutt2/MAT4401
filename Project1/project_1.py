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

def midpoint(): #Define midpoint method
   while(True):
      a = input('Please enter the lower bound of the interval: ')
      b = input('Please enter the upper bound of the interval: ')
      a = np.float(a) #Convert to floats
      b = np.float(b)#Convert to floats
      found = False
      i = 0
      fa = np.exp(-a/5) - np.sin(a) #Find f(a)
      fb = np.exp(-b/5) - np.sin(b) #Find f(b)
      if fa*fb < 0: #Check for sign change
         False
         break
      else:
         print "There is more than one or no sign change, please choose a different interval"
   c = 0.0
   fc = 0.0
   start = time.time()
   while(found == False):
      i = i+1
      c = (a+b)/2 #Find midpoint
      fc = np.exp(-c/5) - np.sin(c)
      if abs(fc)<10**-7: #Check if close to zero
         print("Done")
         found = True
      elif fa*fc>0: #Check which side the sign change is on
         a = c
         fa = np.exp(-a/5) - np.sin(a)
      elif fa*fc<0:
         b = c
         fb = np.exp((-b/5)) - np.sin(b)
   end = time.time()
   print("The zero is: " + str(c))
   print("It took " + str(i) + " interations and " + str(end-start) + " seconds.")
   return [c,i] #return data points

def newtons(): #Implement Newton's method
   a = input('Please enter a starting value: ')
   x = sympy.Symbol('x') #Make x symbolic
   start = time.time() 
   y = sympy.exp((-x/5)) - sympy.sin(x) #define f(x)
   f= sympy.lambdify(x, y, 'numpy') #make it a function
   yprime = y.diff(x) #differentiate it
   fprime = sympy.lambdify(x, yprime, 'numpy') #make derivative a function
   found = False
   i = 0
   while(found == False): #Loop till zero is found
      i = i+1
      xk = a - (f(a)/fprime(a)) #find Xn+1
      if abs(f(xk))<10**-7: #Check if close to zero
         print("Done")
         found = True
      else:
         a = xk
   end = time.time()
   print("The zero is: " + str(xk))
   print("It took " + str(i) + " interations and " + str(end-start) + " seconds.")

def secant(): #Implement Secant method
   b = input('Please enter the first starting value: ')
   a = input('Please enter the second starting value: ')
   b = np.float(b) #Make floats from values
   a = np.float(a) #Make floats from values
   x = sympy.Symbol('x') #Make x symbolic
   start = time.time()
   y = sympy.exp((-x/5)) - sympy.sin(x) #make f(x)
   f = sympy.lambdify(x, y, 'numpy') #Make it a function
   found = False
   i = 0
   while(found == False):
      i = i+1
      xk = a - (f(a)*(a-b))/(f(a)-f(b)) #find Xn+1
      if abs(f(xk))<10**-7:#Check if close to zero
         print("Done")
         found = True
      else:
         b = a
         a = xk
   end = time.time()
   print("The zero is: " + str(xk))
   print("It took " + str(i) + " interations and " + str(end-start) + " seconds.")

def visual(): #Define visual graph
   def onclick(event): #Define pop up for when graph is clicked
    msg = "Do you want to choose x=%f"%(event.xdata)
    title = "Please Confirm"
    if eg.ynbox(msg, title, choices=('Yes', 'No'), image=None):     # show a Continue/Cancel dialog
       print 'x=%f'%(event.xdata)
       print 'The difference is: ' + str( np.sin(event.xdata) - np.exp(-event.xdata/5))
       pyplot.close()  # user chose Continue
   
   a = input('Please enter the lower bound of the interval to graph: ')
   b = input('Please enter the upper bound of the interval to graph: ')
   print("Please click on the point you want to choose for intersection")
   a = np.float(a) #Make floats from input
   b = np.float(b) #Make floats from input
   x1 = np.arange(a,b,.01)
   y1 = np.sin(x1)  #set y1 to sin function
   y2 = np.exp(-x1/5) #set y2 to exp function
   fig = pyplot.figure()
   ax = fig.add_subplot(111) #define 2d plot
   ax.plot( x1,y1 )
   ax.plot( x1,y2 ) #add to plot
   ax.axis([a, b, -1.5, 1.5]) #Set axis
   cid = fig.canvas.mpl_connect('button_press_event', onclick)
   pyplot.title( 'Plotting sin(x) and e^(-x/5)' )
   pyplot.xlabel( 'X Axis' )
   pyplot.ylabel( 'Y Axis' )
   pyplot.show() #show plot
       

def lagrange(): #define Lagrange polynomial
   n = input('What order lagrange polynomial do you want?: ')
   points=[] 
   for i in np.arange(-5.,5.,(10./(n))): #make points from order
      points.append([i,1./(1.+i**2.)])
   points.append([5.,1./(1.+5.**2.)])
   P = 0
   start = time.time()
   for i in points: #loop through all points
      ls = points[:] #make copy of points list
      ls.remove(i) #remove current point form list
      L  = 1
      for k in ls: # make L(x)
         x = sympy.Symbol('x') #make x symbolic
         y = (x-k[0])/(i[0]-k[0]) #create y(x)
         L = L * y
      P = P + (i[1]*L) #make P(x)
   end = time.time()
   Pn = sympy.simplify(P) #Make it readable
   print "The lagrange Polynomail is: " + str(Pn)
   print "It took  " + str(end-start) + " seconds to calculate"
   Pu = sympy.lambdify(x, P, 'numpy')
   Pe = i[1]- Pu(i[0])
   print "The error is: " +  str(Pe)
   x = np.arange(-5.,5.,(10./(n)))
   xlot =  np.linspace(-5,5,1000) #Graph lagrange polynomial
   y = Pu(xlot)
   z = 1/(1+xlot**2) 
   zpoint = 1/(1+x**2)
   pyplot.plot(xlot,y,label="Lagrange Polynomial")
   pyplot.plot(x,zpoint,'co')
   pyplot.plot(xlot,z,label="1/(1+x^2)") #Add original function to graph
   pyplot.legend()
   pyplot.ylim(0,2)
   pyplot.title(str(n)+"th Degree Lagrange Polynomial")
   pyplot.show()

def lform(i): #Functiont to create L(x) for lagrange2
      ls = data[:]
      ls.remove(i)
      L = 1
      for k in ls:
         x = sympy.Symbol('x')
         y = (x-k[0])/(i[0]-k[0])
         L = L * y
      return(i[0]*L)

def lagrange2(): #Optimized Lagrange for multicore/multiple processor system.
   n = input('What order lagrange polynomial do you want?: ')
   points=[]
   for i in np.arange(-5.,5.,(10./(n))): #Make points from order
      points.append([i,1./(1.+i**2.)])
   points.append([5.,1./(1.+5.**2.)])
   global data
   data = points[:]
   P = 0
   start = time.time()
   queue1 = Queue() #create a queue object
   procs = []
   pool = Pool(processes=24)              # start 24 worker processes, set this to # of your cores
   result = pool.map(lform, points) #Create and save f(x)L(x)'s to results
   x = sympy.Symbol('x')
   P = sum(result) #Make P(x)'s
   end = time.time()
   Pn = sympy.simplify(P) #Make it readable
   print "The lagrange Polynomail is: " + str(Pn)
   print "It took  " + str(end-start) + " seconds to calculate"
   Pu = sympy.lambdify(x, P, 'numpy')
   Pe = points[len(points)-1][1]- Pu(points[len(points)-1][0])
   print "The error is: " +  str(Pe)
   x = np.arange(-5.,5.,(10./(n))) #Plot Lagrange and original function
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

def piecewise(): #Define piecewise function
   n = input('How many points do you want? ')
   points = []
   yvals=[]
   xvals=[]
   for i in np.arange(-5.,5.,(10./(n))): #Make points from number of points asked for.
      points.append([i,1./(1.+i**2.)])
      yvals.append(1./(1.+i**2.))
      xvals.append(i)
   yvals.append(1./(1.+5.**2.))
   xvals.append(5.)
   yvals = np.array(yvals)#make list to array
   xvals = np.array(xvals)#make list to array
   points.append([5.,1./(1.+5.**2.)])
   pw = []
   start = time.time()
   for k in points: #Loop through all points
      if 0 == points.index(k): #skip first point
         j = k
      else:
         xsy = sympy.Symbol('x') #make x symbolic
         y = ((k[1]-j[1])/(k[0]-j[0]))*(xsy-j[0]) - j[1] #find y(x)
         pw.append([y,j[0],k[0]]) #at it to list along with bounds
         j = k
   end = time.time()
   print "the piecewise function is as follows: "
   print "F(x) = { " #output piecewise function
   for i in pw:
      print str(i[0]) + "      for " + str(i[1]) + " < x < " + str(i[2]) 
   print "}"
   print "It took  " + str(end-start) + " seconds to calculate"
# Plot the piece wise function, the points and the original function
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
   

def raised_cosine(): #Define raised cosine function
   n = input('How many points do you want? ')
   points = []
   for i in np.arange(-5.,5.,(10./(n))): #Make points
      points.append([i,1./(1.+i**2.)])
   points.append([5.,1./(1.+5.**2.)])
   pw = []
   j = points[0]
   x = sympy.Symbol('x') #make x symbolic
   pw = []
   start = time.time()
   for k in points: #loop through all points
      if 0 == points.index(k):
         j = k
      else: 
         mu = j[0]+(x-j[0])/(k[0]-j[0]) #define mu
         mu2 = (1-sympy.cos(mu*np.pi))/2 #define mu2, the cosine funcytion
         y = (j[1]*(1-mu2)+k[1]*mu2) #finally create y(x)
         pw.append([y,j[0],k[0]]) #add y(x) and bounds to list
         j = k
   end = time.time()
   for i in range(len(pw)):
    if i%2==0:#check for odd entry
        print pw[i][0] 
        W = sympy.lambdify(x,pw[i][0],'numpy')
        pw[i][0]  = -(pw[i][0].as_leading_term(x) - W(.5))*sympy.cos(np.pi*x)+W(.5) #change sign on every other piecewise cosine function, it got flipped earlier
        print pw[i][0]
   print "the piecewise function is as follows: " #Output piecewise function
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

def lsa(): #Define least square approximation
   n = input('What order least square approximation polynomial do you want?: ')
   points=[]
   yvals=[]
   xvals=[]
   start = time.time()
   for i in np.arange(-5.,5.,(10./(n))): #make points
      points.append([i,1./(1.+i**2.)])
      yvals.append(1./(1.+i**2.))
      xvals.append(i)
   points.append([5.,1./(1.+5.**2.)])
   yvals.append(1./(1.+5.**2.))
   xvals.append(5.)
   yvals = np.array(yvals) #Make arrays
   xvals = np.array(xvals) #Make arrays
   A = np.vander(xvals, n) #Make matrix
   # find the x that minimizes the norm of Ax-y
   (coeffs, residuals, rank, sing_vals) = np.linalg.lstsq(A, yvals)
   # create a polynomial using coefficients
   f = np.poly1d(coeffs) #Make polynomial
   end = time.time()
   print "The least square approximation is: " + str(f)
   print "It took  " + str(end-start) + " seconds to calculate"
   ss_err=(residuals**2).sum() #Find R^2
   ss_tot=((yvals-yvals.mean())**2).sum()
   rsquared=1-(ss_err/ss_tot)
   print("The rsquared value is: " + str(rsquared))
   x = np.arange(-5.,5.,(10./(n))) #Plot function along with original function
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

def interpolate(): #Define Interpolation menu
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

def exit(): #Define Exit
   return True

def root(): #Define root finding menu
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

def main_menu(): #Define main menu
  while(True):
   print("Please choose one of the following options:")
   print("1. Root finding")
   print("2. Interpolation")
   print("3. Quit")
   n = input("")
   if(mmenu[n]()):
      break
#Define dictionaries of menus
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
main_menu() #run main menu

